"""Real CUDA expert arithmetic stays native for one-token requests."""
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch=None
if torch is not None:
    from utils.eval_core import DistributedSparseMoeBlock


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'CUDA required')
class NativeExpertGranularityTest(unittest.TestCase):
    def test_ep_and_staged_expert_outputs_match_separate_native_requests(self):
        torch.manual_seed(13)
        device=torch.device('cuda:0')
        hidden=torch.randn(2,2048,device=device,dtype=torch.float16)
        weights=(torch.randn(1408,2048,device=device,dtype=torch.float16)/32,
                 torch.randn(1408,2048,device=device,dtype=torch.float16)/32,
                 torch.randn(2048,1408,device=device,dtype=torch.float16)/32)
        block=SimpleNamespace(_native_decode_granularity=True,_current_stage=lambda:'decode',
            _current_forward_batch_size=1,_current_forward_seq_len=1,act_fn=torch.nn.functional.silu,
            experts_kind='modulelist',expert_offload='cpu',_get_local_module_weights=lambda eid:weights)
        block._module_expert_forward=lambda *args:DistributedSparseMoeBlock._module_expert_forward(block,*args)
        gw,uw,dw=weights
        with torch.inference_mode():
            native=torch.cat([torch.nn.functional.linear(torch.nn.functional.silu(
                torch.nn.functional.linear(row,gw))*torch.nn.functional.linear(row,uw),dw)
                for row in hidden.split(1)],dim=0)
            route_weights=torch.tensor([[.23],[.41]],device=device,dtype=torch.float16)
            for function in (DistributedSparseMoeBlock._plain_ep_compute_segment,
                             DistributedSparseMoeBlock._epipe_compute_segment):
                extra={'payload_override':weights} if function.__name__=='_epipe_compute_segment' else {}
                actual,_=function(block,eid=0,cur=hidden,cur_w=route_weights,eid_is_local=True,
                                  allow_nonlocal=False,**extra)
                torch.testing.assert_close(actual,native*route_weights,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
