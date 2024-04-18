import argparse
import torch.distributed as dist
from common import _GLOBAL_ARGS

def finetune():
   pass
   
def initialize(rank, world_size, args):
   # Cast and role
    context = Context(rank=rank, world_size=world_size,
                      cast={'worker': (0, world_size-1)},
                      config=config, output_path=args.output_path)
    role = context.role

    # Initialize logging
    log_file = os.path.join(args.log_path, '{}_rank_{}.log'.format(role, rank))
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='[%(asctime)s] [%(levelname)s][%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', force=True)

   master_addr = os.environ['MASTER_ADDR']
   master_port = os.environ['MASTER_PORT']
   init_method = 'tcp://{}:{}'.format(master_addr, int(master_port))
   dist.init_process_group(
      backend="gloo", rank=context.rank, world_size=context.world_size, init_method=init_method)
   
   finetune(context, args)
   
if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="""
                                    A demo for llama fine-tuning with gradient checkpoint, 
                                    mixed precision, data parallel, tensor parallel.
                                    """
                                 ) 
   parser.add_argument("--use_dp", help="use data parallel", action="store_true")
   parser.add_argument("--use_tp", help="use tensor parallel", action="store_true")
   parser.add_argument("--use_amp", help="use mixed precesion", action="store_true")
   parser.add_argument("--use_grad_ckpt", help="use gradient checkpoint", action="store_true")
   _GLOBAL_ARGS = parser.parse_args()

   initialize(rank, world_size, args)

   
# torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 train.py