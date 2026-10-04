import importlib.util
import sys
import types
import unittest
from pathlib import Path
import torch
import gradio as gr

root = Path(__file__).parent
modules = types.ModuleType('modules')
modules.scripts = types.SimpleNamespace(ScriptBuiltinUI=object, AlwaysVisible=True)
modules.shared = types.SimpleNamespace(cmd_opts=types.SimpleNamespace(lora_dir=None, lora_dirs=[]))
sys.modules['modules'] = modules
ui = types.ModuleType('modules.ui_components')
class InputAccordion:
    def __init__(self, value, label, **kwargs): self.value, self.label, self.kwargs = value,label,kwargs
    def __enter__(self):
        self.group=gr.Group(**self.kwargs); self.group.__enter__()
        return gr.Checkbox(value=self.value,label=self.label)
    def __exit__(self,*args): return self.group.__exit__(*args)
ui.InputAccordion=InputAccordion
sys.modules['modules.ui_components']=ui
script_path = root/'krea2_rebalance.py'
if not script_path.exists(): script_path = root.parent/'scripts'/'krea2_rebalance.py'
spec=importlib.util.spec_from_file_location('krea_under_test',script_path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Residual(torch.nn.Module):
    def forward(self,x,**kwargs): return x.add_(torch.linspace(-.3,.3,x.shape[-1],device=x.device,dtype=x.dtype))
class Attention(torch.nn.Module):
    def __init__(self): super().__init__();self.wv=torch.nn.Identity()
    def forward(self,x): return self.wv(x).mean(dim=1,keepdim=True).expand_as(x)*.1
class Block(torch.nn.Module):
    def __init__(self): super().__init__();self.attn=Attention()
    def forward(self,x,**kwargs): return x + self.attn(x)
class Fusion(torch.nn.Module):
    __module__='backend.nn.krea'
    def __init__(self):
        super().__init__();self.layerwise_blocks=torch.nn.ModuleList([Residual(),Residual()]);self.refiner_blocks=torch.nn.ModuleList([Block()]);self.calls=0
    def forward(self,x,mask=None,transformer_options=None):
        self.calls+=1
        b,s,t,d=x.shape;y=x.reshape(b*s,t,d)
        for block in self.layerwise_blocks:y=block(y.contiguous())
        y=y.reshape(b,s,t,d).mean(2)
        for block in self.refiner_blocks:y=block(y)
        return y
class DM(torch.nn.Module):
    __module__='backend.nn.krea'
    def __init__(self):
        super().__init__();self.txtfusion=Fusion();self.txtmlp=torch.nn.Identity();self.blocks=torch.nn.ModuleList([Block()]);self.txtlayers=12;self.txtdim=2560
    def forward(self,input,timestep,c_crossattn):
        text=self.txtfusion(c_crossattn)
        combined=torch.cat((text,input),dim=1)
        for block in self.blocks:combined=block(combined)
        return combined[:,text.shape[1]:]
class Patcher:
    def __init__(self,dm=None):self.dm=dm or DM();self.model_options={};self.patches={}
    def get_model_object(self,key):return self.dm
    def clone(self):
        c=Patcher(self.dm);c.model_options=self.model_options.copy();c.patches={k:list(v) for k,v in self.patches.items()};return c
    def set_model_unet_function_wrapper(self,w):self.model_options['model_function_wrapper']=w

def args():return {'input':torch.ones(1,1,2560),'timestep':torch.tensor([.5]),'c':{'c_crossattn':torch.randn(1,3,12,2560)}}
class Tests(unittest.TestCase):
    def setUp(self):torch.manual_seed(4)
    def test_input_preserved(self):
        f=Fusion();x=torch.randn(1,3,12,2560);old=x.clone()
        m._enhanced_txtfusion_forward(f,x,original_forward=f.forward)
        self.assertTrue(torch.equal(x,old))
    def test_zero_matches_baseline_one_pass(self):
        f=Fusion();x=torch.randn(1,3,12,2560);expected=f(x.clone());f.calls=0
        result=m._enhanced_txtfusion_forward(f,x,strength=0,original_forward=f.forward)
        self.assertTrue(torch.equal(result,expected));self.assertEqual(f.calls,1)
    def test_each_layer_uses_one_gain_for_both_1280_halves(self):
        gains=m._chunk_gains(torch.device('cpu'),torch.float32,1).reshape(12,2)
        self.assertTrue(torch.equal(gains[:,0],gains[:,1]))
        self.assertTrue(torch.equal(gains[:,0],torch.tensor(m.ENHANCER_PROFILE_12)))
    def test_low_strength_changes_smoothly_without_early_saturation(self):
        f=Fusion();x=torch.randn(1,3,12,2560);ref=f(x.clone()).float();ratios=[]
        for strength in (.01,.05,.1):
            out=m._enhanced_txtfusion_forward(f,x,strength=strength,token_cap=.25,original_forward=f.forward).float()
            ratios.append(float((m._rms(out-ref)/m._rms(ref)).mean()))
        self.assertLess(ratios[0],ratios[1]);self.assertLess(ratios[1],ratios[2])
        self.assertLess(ratios[1],.1)
        self.assertGreater(ratios[2]-ratios[1],.5*(ratios[1]-ratios[0]))
    def test_noncontiguous(self):
        f=Fusion();x=torch.randn(1,3,2560,12).transpose(-1,-2);old=x.clone()
        result=m._enhanced_txtfusion_forward(f,x,original_forward=f.forward)
        self.assertEqual(result.shape,(1,3,2560));self.assertTrue(torch.equal(x,old))
    def test_joint_limits(self):
        f=Fusion();x=torch.randn(1,3,12,2560);ref=f(x.clone())
        out=m._enhanced_txtfusion_forward(f,x,strength=3,power=3,token_cap=.5,
            negpip={'enabled':True,'weights':[.2]*12,'neg_strength':3,'token_cap':3},original_forward=f.forward)
        self.assertTrue(torch.all(m._rms(out-ref)<=m._rms(ref)*.50001))
        self.assertTrue(torch.all(m._rms(out)<=m._rms(ref)*1.00001))
    def test_nonfinite_candidate_falls_back(self):
        f=Fusion();x=torch.randn(1,3,12,2560);ref=f(x.clone());count=[0];stats={}
        def original(v,**kw):
            count[0]+=1
            return f(v) if count[0]==1 else torch.full_like(ref,float('inf'))
        out=m._enhanced_txtfusion_forward(f,x,original_forward=original,stats=stats)
        self.assertTrue(torch.allclose(out,ref));self.assertEqual(int(stats['invalid_tokens']),3)
    def test_invalid_baseline_raises(self):
        f=Fusion();x=torch.full((1,3,12,2560),float('nan'))
        with self.assertRaises(RuntimeError):m._enhanced_txtfusion_forward(f,x,original_forward=f.forward)
    def test_legacy_zero_skips(self):
        f=Fusion();x=torch.randn(1,3,12,2560)
        m._enhanced_txtfusion_forward(f,x,strength=0,negpip={'enabled':True,'neg_strength':0},original_forward=f.forward)
        self.assertEqual(f.calls,1)
    def test_wrapper_cleanup_on_failure(self):
        dm=DM();w=m._make_unet_wrapper(dm,1,avoid_context=torch.ones(1,2,12,2560));original=dm.txtfusion.forward
        with self.assertRaises(RuntimeError):w(lambda *a,**k: (_ for _ in ()).throw(RuntimeError('fixture')),args())
        self.assertEqual(dm.txtfusion.forward,original);self.assertNotIn('forward',dm.txtfusion.__dict__)
        self.assertTrue(all(not x._forward_hooks for x in dm.modules()))
    def test_previous_wrapper_preserved(self):
        called=[];dm=DM()
        def previous(fn,kw):called.append(True);return m._call_model_function(fn,kw)
        m._make_unet_wrapper(dm,1,previous)(dm,args());self.assertEqual(called,[True])
    def test_wrapper_does_not_stack(self):
        dm=DM();w=m._make_unet_wrapper(dm,1);w2=m._make_unet_wrapper(dm,1,w)
        self.assertIsNone(w2._krea2_previous)
    def test_cache_and_invalidation(self):
        dm=DM();w=m._make_unet_wrapper(dm,1,use_cache=True);kw=args();a=w(dm,kw);count=dm.txtfusion.calls
        b=w(dm,kw);self.assertEqual(dm.txtfusion.calls,count);self.assertTrue(torch.equal(a,b));self.assertEqual(w._krea2_stats['cache_hits'],1)
        kw['c']['c_crossattn'].add_(.1);w(dm,kw);self.assertGreater(dm.txtfusion.calls,count)
    def test_cache_output_owned(self):
        dm=DM();w=m._make_unet_wrapper(dm,1,use_cache=True);kw=args();a=w(dm,kw);a.zero_();b=w(dm,kw)
        self.assertFalse(torch.equal(a,b))
    def test_foreign_hook_disables_cache(self):
        dm=DM();handle=dm.txtfusion.refiner_blocks[0].attn.wv.register_forward_hook(lambda mod,a,out:out)
        w=m._make_unet_wrapper(dm,1,use_cache=True);kw=args();w(dm,kw);w(dm,kw)
        self.assertEqual(w._krea2_stats.get('cache_hits',0),0);handle.remove()
    def test_selective_slice_only(self):
        dm=DM();x=torch.ones(1,7,2560)
        with m._negative_value_hooks(dm,3,5):
            actual=dm.blocks[0].attn.wv(x)
            self.assertTrue(torch.equal(actual[:,:3],x[:,:3]));self.assertTrue(torch.equal(actual[:,3:5],-x[:,3:5]));self.assertTrue(torch.equal(actual[:,5:],x[:,5:]))
        self.assertTrue(torch.equal(dm.blocks[0].attn.wv(x),x))
    def test_avoid_changes_output_preserves_context(self):
        dm=DM();kw=args();old=kw['c']['c_crossattn'].clone()
        a=m._make_unet_wrapper(dm,0)(dm,kw);b=m._make_unet_wrapper(dm,0,avoid_context=torch.ones(1,2,12,2560))(dm,kw)
        self.assertFalse(torch.equal(a,b));self.assertEqual(a.shape,b.shape);self.assertTrue(torch.equal(old,kw['c']['c_crossattn']))
    def test_numeric_validation(self):
        self.assertIsNone(m._parse_floats('nan'));self.assertIsNone(m._parse_floats('4'));self.assertEqual(m._bounded_float(float('inf'),1,0,3),1)
    def test_lookup_formats_duplicates(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            modules.shared.cmd_opts.lora_dir=d;Path(d,'adapter.txt').touch();self.assertIsNone(m._find_lora_file('adapter'))
            Path(d,'adapter.safetensors').touch();self.assertIsNotNone(m._find_lora_file('adapter'))
            Path(d,'nested').mkdir();Path(d,'nested','adapter.safetensors').touch()
            with self.assertRaises(ValueError):m._find_lora_file('adapter')
        modules.shared.cmd_opts.lora_dir=None
    def test_ui_and_reset_callback(self):
        with gr.Blocks() as demo:controls=m.Krea2RebalanceScript().ui()
        self.assertEqual(len(controls),15)
        self.assertFalse(controls[1].value)
        self.assertEqual(controls[3].value,.15)
        self.assertEqual(controls[6].value,.05)
        self.assertEqual(controls[6].maximum,m.TXTFUSION_TOKEN_HARD_CAP)
        self.assertFalse(controls[-2].value)
        self.assertEqual(controls[-1].value,.35)
        reset=next(fn for fn in demo.fns.values() if len(fn.outputs)==16)
        values=reset.fn();self.assertEqual(len(values),len(reset.outputs));self.assertEqual(values[0],.15);self.assertEqual(values[2],.05);self.assertFalse(values[3]);self.assertFalse(values[11]['visible']);self.assertEqual(values[13],'');self.assertFalse(values[14])
        presets=sorted((fn.fn()[0],fn.fn()[2]) for fn in demo.fns.values() if len(fn.outputs)==4 and not fn.inputs)
        self.assertEqual(presets,[(.15,.05),(.35,.1),(.65,.15)])

    def test_old_cap_is_safely_clamped_and_recorded(self):
        base=Patcher();p=types.SimpleNamespace(sd_model=types.SimpleNamespace(forge_objects=types.SimpleNamespace(unet=base)),extra_generation_params={})
        script=m.Krea2RebalanceScript();script.process_before_every_sampling(p,True,False,'',1,token_cap=1.3)
        self.assertEqual(p.extra_generation_params['Krea2 Rebalance Requested Adherence Cap'],1.3)
        self.assertEqual(p.extra_generation_params['Krea2 Rebalance Adherence Cap'],m.TXTFUSION_TOKEN_HARD_CAP)
        self.assertEqual(p.sd_model.forge_objects.unet.model_options['model_function_wrapper']._krea2_stats,{})
        script.post_sample(p,None);self.assertIs(p.sd_model.forge_objects.unet,base)
    def test_registered_patch_status(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path=Path(d,'a.safetensors');path.touch();modules.shared.cmd_opts.lora_dir=d
            nets=types.ModuleType('networks');nets.available_networks={};nets.available_network_aliases={};nets.loaded_networks=[];nets.load_lora_state_dict=lambda p:{'x':torch.ones(1)}
            def loader(u,c,data,s,t,**kw):
                result=u.clone();result.patches['txtfusion.weight']=[(s,'patch')];return result,None
            nets.load_lora_for_models=loader;sys.modules['networks']=nets
            p=types.SimpleNamespace(extra_generation_params={});u,status=m._apply_adapter(p,Patcher(),'a',1)
            self.assertTrue(status.startswith('Registered 1'))
            nets.load_lora_for_models=lambda u,*a,**k:(u,None)
            _,status=m._apply_adapter(p,Patcher(),'a',1);self.assertTrue(status.startswith('No model'))
            del sys.modules['networks'];modules.shared.cmd_opts.lora_dir=None

    def test_adapter_does_not_accumulate_between_sampling_passes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path=Path(d,'a.safetensors');path.touch();modules.shared.cmd_opts.lora_dir=d
            nets=types.ModuleType('networks');nets.available_networks={};nets.available_network_aliases={};nets.loaded_networks=[];nets.load_lora_state_dict=lambda p:{'x':torch.ones(1)}
            def loader(u,c,data,s,t,**kw):
                result=u.clone();result.patches.setdefault('txtfusion.weight',[]).append((s,'patch'));return result,None
            nets.load_lora_for_models=loader;sys.modules['networks']=nets
            base=Patcher();p=types.SimpleNamespace(sd_model=types.SimpleNamespace(forge_objects=types.SimpleNamespace(unet=base)),extra_generation_params={})
            script=m.Krea2RebalanceScript();script.process(p)
            script.process_before_every_sampling(p,True,True,'a',1)
            self.assertEqual(len(p.sd_model.forge_objects.unet.patches['txtfusion.weight']),1)
            script.process_before_every_sampling(p,True,True,'a',1)
            self.assertEqual(len(p.sd_model.forge_objects.unet.patches['txtfusion.weight']),1)
            script.post_sample(p,None);self.assertIs(p.sd_model.forge_objects.unet,base);self.assertFalse(base.patches)
            del sys.modules['networks'];modules.shared.cmd_opts.lora_dir=None

    def test_sampling_lifecycle_restores_base(self):
        base=Patcher();p=types.SimpleNamespace(sd_model=types.SimpleNamespace(forge_objects=types.SimpleNamespace(unet=base)),extra_generation_params={})
        script=m.Krea2RebalanceScript();script.process(p)
        script.process_before_every_sampling(p,True,False,'',1)
        first=p.sd_model.forge_objects.unet
        script.process_before_every_sampling(p,True,False,'',1)
        self.assertIs(p._krea2_base_unet,base)
        wrapper=p.sd_model.forge_objects.unet.model_options['model_function_wrapper']
        self.assertIsNone(wrapper._krea2_previous)
        script.post_sample(p,None)
        self.assertIs(p.sd_model.forge_objects.unet,base)
        self.assertFalse(hasattr(p,'_krea2_base_unet'))
    def test_avoid_cfg_guard(self):
        backend_args=types.ModuleType('backend.args');backend_args.dynamic_args=types.SimpleNamespace(ref_latents=[])
        sys.modules['backend.args']=backend_args
        p=types.SimpleNamespace(sd_model=types.SimpleNamespace(forge_objects=types.SimpleNamespace(unet=Patcher())),cfg_scale=2,extra_generation_params={})
        m.Krea2RebalanceScript().process_before_every_sampling(p,True,False,'',1,avoid_text='stripes')
        self.assertEqual(p.extra_generation_params['Krea2 Avoid Status'],'Skipped: Avoid requires CFG 1')
    def test_cache_standard_sampler_options(self):
        dm=DM();w=m._make_unet_wrapper(dm,1,use_cache=True);kw=args()
        def model(inp,t,c_crossattn,transformer_options):
            return dm.txtfusion(c_crossattn,transformer_options=transformer_options)
        kw['c']['transformer_options']={'sigmas':torch.tensor([1.]),'cond_or_uncond':[0],'cond_mark':torch.ones(1),'cond_indices':[0],'uncond_indices':[]}
        a=w(model,kw);kw['c']['transformer_options']['sigmas']=torch.tensor([.5]);b=w(model,kw)
        self.assertTrue(torch.equal(a,b));self.assertEqual(w._krea2_stats['cache_hits'],1)
        kw['c']['transformer_options']['optimized_attention_override']='fixture'
        w(model,kw);self.assertFalse(w._krea2_cache)
    def test_encode_avoid_boundaries(self):
        backend=types.ModuleType('backend');backend.memory_management=types.SimpleNamespace(load_model_gpu=lambda p:None)
        sys.modules['backend']=backend
        import contextlib
        modules.devices=types.SimpleNamespace(autocast=contextlib.nullcontext)
        tokens=[151644,11,151645,151644,872,198,41,42,151645,151644,77]
        engine=types.SimpleNamespace(id_template=151644,tokenize=lambda texts:[tokens],process_tokens=lambda a,b:torch.arange(len(tokens)).reshape(1,1,-1,1).expand(1,12,-1,2560).float())
        p=types.SimpleNamespace(sd_model=types.SimpleNamespace(text_processing_engine_qwen=engine,forge_objects=types.SimpleNamespace(clip=types.SimpleNamespace(patcher=None))))
        out=m._encode_avoid(p,'stripes');self.assertEqual(out.shape,(1,2,12,2560));self.assertEqual(float(out[0,0,0,0]),6.)
        with self.assertRaises(ValueError):m._encode_avoid(p,'(stripes:2)')
    def test_batch_avoid(self):
        dm=DM();kw=args();kw['input']=kw['input'].expand(2,-1,-1);kw['c']['c_crossattn']=kw['c']['c_crossattn'].expand(2,-1,-1,-1)
        out=m._make_unet_wrapper(dm,0,avoid_context=torch.ones(1,2,12,2560))(dm,kw)
        self.assertEqual(out.shape,(2,1,2560));self.assertTrue(all(not mod._forward_hooks for mod in dm.modules()))

    def test_hires_cfg_guard(self):
        backend_args=types.ModuleType('backend.args');backend_args.dynamic_args=types.SimpleNamespace(ref_latents=[])
        sys.modules['backend.args']=backend_args
        p=types.SimpleNamespace(sd_model=types.SimpleNamespace(forge_objects=types.SimpleNamespace(unet=Patcher())),cfg_scale=1,hr_cfg=2,is_hr_pass=True,extra_generation_params={})
        m.Krea2RebalanceScript().process_before_every_sampling(p,True,False,'',1,avoid_text='stripes')
        self.assertEqual(p.extra_generation_params['Krea2 Avoid Status'],'Skipped: Avoid requires CFG 1')
    def test_replace_wrapper_with_zero_strength(self):
        base=Patcher();base.model_options['model_function_wrapper']=m._make_unet_wrapper(base.dm,1)
        p=types.SimpleNamespace(sd_model=types.SimpleNamespace(forge_objects=types.SimpleNamespace(unet=base)),extra_generation_params={})
        m.Krea2RebalanceScript().process_before_every_sampling(p,True,False,'',0)
        self.assertNotIn('model_function_wrapper',p.sd_model.forge_objects.unet.model_options)

if __name__=='__main__':unittest.main(verbosity=2)
