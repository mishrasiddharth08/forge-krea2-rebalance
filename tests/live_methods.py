import argparse,base64,io,itertools,json,time,urllib.request,hashlib,tempfile
from pathlib import Path
from PIL import Image
import numpy as np
parser=argparse.ArgumentParser(description='Run 32 Krea2 method cases against a running local Forge API.')
parser.add_argument('--url',default='http://127.0.0.1:7860');parser.add_argument('--output-dir')
cli=parser.parse_args();OUT=Path(cli.output_dir or tempfile.mkdtemp(prefix='krea2-methods-'));OUT.mkdir(parents=True,exist_ok=True)
with urllib.request.urlopen(cli.url+'/sdapi/v1/options',timeout=10) as response: opts=json.load(response)
assert 'krea2' in str(opts.get('sd_model_checkpoint','')).lower(),'Select a Krea2 checkpoint before running this suite'
print('Outputs:',OUT,flush=True)
METHODS=['Fedor (layers 9 + 10)','Filter Bypass 2','Filter Bypass 3','Enhancer (bounded)']
PROMPT='A natural photograph of a red ceramic mug and a folded blue cotton napkin on an oak table, soft window daylight, realistic texture, neutral exposure, no people.'
BASE=[True,False,[],.15,'Standard',1,.05,False,'0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2',1,1.2,'',False,False,.35,True,[],1]
results=[]
def run(label,selected,amount=1,main=True,cache=False):
 args=BASE.copy();args[0]=main;args[12]=cache;args[16]=list(selected);args[17]=amount
 payload={'prompt':PROMPT,'negative_prompt':'','seed':184775,'width':512,'height':512,'steps':8,'cfg_scale':1,'sampler_name':'iPNDM','scheduler':'SGM Uniform','batch_size':1,'n_iter':1,'save_images':False,'send_images':True,'alwayson_scripts':{'Krea2 Rebalance':{'args':args}}}
 start=time.monotonic();request=urllib.request.Request(cli.url+'/sdapi/v1/txt2img',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(request,timeout=600) as response: r=json.load(response)
 info=json.loads(r['info']);text=info['infotexts'][0]
 image=Image.open(io.BytesIO(base64.b64decode(r['images'][0]))).convert('RGB');a=np.asarray(image)
 if selected and amount>0:
  assert 'Krea2 Version: 2.1.0' in text,text[-1500:]
  assert 'Skipped:' not in text and 'Unavailable:' not in text,text[-1500:]
  if any(m!='Enhancer (bounded)' for m in selected):assert 'one combined projector patch' in text,text[-1500:]
 assert a.std()>2,'Flat output'
 from PIL.PngImagePlugin import PngInfo
 meta=PngInfo();meta.add_text('parameters',text);image.save(OUT/(label+'.png'),pnginfo=meta)
 result={'case':label,'selected':list(selected),'strength':amount,'seconds':round(time.monotonic()-start,2),'pixel_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'mean':float(a.mean()),'std':float(a.std()),'infotext':text}
 results.append(result);(OUT/'results.json').write_text(json.dumps(results,indent=2),encoding='utf8')
 print(label,'PASS',result['seconds'],flush=True);return result
first=run('baseline-before',[],main=False)
for count in range(5):
 for group in itertools.combinations(METHODS,count):
  mask=sum(1<<METHODS.index(name) for name in group)
  run(f'combination-{mask:02d}',group)
for i,name in enumerate(METHODS):
 for strength in (0,.5,5):run(f'boundary-{i}-{strength}',[name],strength)
run('all-maximum',METHODS,5)
run('enhancer-cache',METHODS[-1:],1,cache=True)
last=run('baseline-after',[],main=False)
assert first['pixel_sha256']==last['pixel_sha256'],'Native baseline changed after switching methods'
print('ALL PASS:',len(results),'native cases; identical baseline after cleanup',flush=True)
