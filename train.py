import argparse


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

   