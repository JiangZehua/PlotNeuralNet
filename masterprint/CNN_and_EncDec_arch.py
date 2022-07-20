import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *

width = 30
im_w = width / 5

arch = [ 
    to_head('..'), 
    to_cor(),
    to_begin(),
    
    to_Latent( name='input', n_filer=16, s_filer=4, offset="(-3.5,0,0)", width=4, height=width/4, depth=width/4),
    # to_input( 'frame_2149.png', name='input', width=im_w, height=im_w),
    to_DeConvRelu( name='l1', s_filer=8, n_filer=32, width=8, caption="3$\\times $3; 2-stride---ReLu", height=width/2, depth=width/2, offset="(-0.7,0,0)"),
    to_connection( "input", "l1" ),
    to_DeConvRelu( name='l2', s_filer=16, n_filer=32, width=8, offset="(3.2,0,0)", caption="3$\\times$3; 2-stride---ReLu", height=width, depth=width),
    to_connection( "l1", "l2" ),
    to_ConvSigmoid( name='l3', s_filer=16, n_filer=9, width=2.5, offset="(7.1,0,0)", caption="1$\\times$1---Sigmoid", height=width, depth=width),
    to_connection( "l2", "l3" ),
    to_Onehot( name='output', n_filer=9, s_filer=16, offset="(9.9,0,0)", width=2.5, height=width, depth=width),
    to_connection( "l3", "output" ),
    to_input( 'generated_print_0.png', to='(10.4,0,0)', width=im_w, height=im_w),
    to_end() 
    ]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()