import torch
from test_krea2 import m,Fusion
assert torch.cuda.is_available()
for dtype in (torch.float16,torch.bfloat16):
    torch.manual_seed(42)
    x=torch.randn(1,8,12,2560,device='cuda',dtype=dtype);old=x.clone();f=Fusion().cuda()
    with torch.inference_mode():
        ref=f(x.clone())
        out=m._enhanced_txtfusion_forward(f,x,strength=1.2,power=1.5,token_cap=.8,
                negpip={'enabled':True,'weights':[.2]*12,'neg_strength':1,'token_cap':1.2},original_forward=f.forward)
    assert torch.equal(x,old)
    assert torch.isfinite(out).all()
    ratio=(m._rms(out.float()-ref.float())/m._rms(ref.float())).max().item()
    assert ratio <= .81, ratio
    print(str(dtype), 'PASS: CUDA input preservation, finite output, combined limit',round(ratio,5))
