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
    
    to_LatentVector( name='input', n_filer=100, caption='z', offset="(-4.5,0,0)", width=1, height=6, depth=1),
    # to_input( 'generated_print_0.png', name='input', width=im_w, height=im_w),
    to_LatentVector( name='dense', n_filer=32768, offset="(-3,0,0)", width=1, height=15, depth=1),
    to_connection( "input", "dense" ),
    to_DeConvRelu( name='l1', s_filer=32, n_filer=32, width=10, caption="32$\\times $32$\\times$32; LeakyReLu", height=width/3.5, depth=width/3.5, offset="(-0.7,0,0)"),
    to_connection( "dense", "l1" ),
    to_DeConvRelu( name='l2', s_filer=64, n_filer=64, width=20, offset="(3.2,0,0)", caption="64$\\times $64$\\times$64; LeakyReLu", height=width/1.5, depth=width/1.5),
    to_connection( "l1", "l2" ),
    to_Conv( name='l3', s_filer=128, n_filer=1, width=2.5, offset="(9.5,0,0)", caption="128$\\times$128$\\times$1", height=width, depth=width),
    to_connection( "l2", "l3" ),
    to_Onehot( name='output', n_filer=1, s_filer=128, offset="(12,0,0)", caption="Output", width=2.5, height=width, depth=width),
    to_connection( "l3", "output" ),
    to_input( 'fake_print.png', to='(12.5,0,0)', width=im_w, height=im_w),
    to_end() 
    ]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()