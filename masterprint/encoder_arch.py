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
    
    to_Onehot( name='input', n_filer=1, s_filer=128, offset="(-3.5,0,0)", caption="Input", width=2.5, height=width, depth=width),
    to_input('real_print.png', name='input', width=im_w, height=im_w),
    to_ConvLeakyRelu( name='l1', s_filer=64, n_filer=64, width=20, caption="64$\\times $64$\\times$64; LeakyReLu", height=width/1.5, depth=width/1.5, offset="(-0.7,0,0)"),
    to_connection( "input", "l1" ),
    to_ConvLeakyRelu( name='l2', s_filer=32, n_filer=32, width=10, offset="(5.2,0,0)", caption="32$\\times $32$\\times$32; LeakyReLu", height=width/3.5, depth=width/3.5),
    to_connection( "l1", "l2" ),
    to_LatentVector( name='flatten', n_filer=32768, width=1, offset="(9.1,0,0)", caption="Flatten", height=15, depth=1),
    to_connection( "l2", "flatten" ),
    to_LatentVector( name='mean', n_filer=100, offset="(11.9,2,0)", width=1, caption="mean", height=6, depth=1),
    to_connection( "flatten", "mean" ),
    to_LatentVector( name='var', n_filer=100, offset="(11.9,-2,0)", width=1, caption="var", height=6, depth=1),
    to_connection( "flatten", "var" ),
    to_LatentVector( name='output', n_filer=100, offset="(13.9,0,0)", caption='z', width=1, height=6, depth=1),
    to_connection( "mean", "output" ),
    to_connection( "var", "output" ),
    to_end() 
    ]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()