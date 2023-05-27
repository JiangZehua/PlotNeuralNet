
import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *

width = 30
im_w = width / 5

fc_height = 60
fc_width = 1

arch = [ 
    to_head('..'), 
    to_cor(),
    to_begin(),

    ###### feature ######
    to_Onehot( name='input', n_filer=3, s_filer=32, offset="(-3.5,0,0)", width=2.5, height=width, depth=width),
    to_input( 'test_img.png', name='input', width=im_w/2, height=im_w/2),
    to_ConvRelu( name='l1', s_filer=32, n_filer=64, width=16, caption="Conv2D-7$\\times$7--ReLu", height=width, depth=width, offset="(-0.5,0,0)"),
    to_connection( "input", "l1" ),
    to_ConvRelu( name='l2', s_filer=16, n_filer=64, width=16, offset="(4.7,0,0)", caption="Conv2D-7$\\times$7--ReLu", height=width/2, depth=width/2),
    to_connection( "l1", "l2" ),
    # to_Vector( name='l3', n_filer=4096, width=fc_width, offset="(9.1,0,0)", caption="FC1", height=fc_height*2, depth=fc_width),
    to_Conv1DRelu( name='l3', s_filer=1, n_filer=4096, offset="(9.1,0,0)", caption="FC1--ReLu", width=2, height=fc_height*1.5, depth=fc_width),
    to_connection( "l2", "l3" ),
    to_Conv1DRelu( name='l4', s_filer=1, n_filer=256, offset="(11.1,0,0)", caption="FC2--ReLu", width=2, height=fc_height, depth=fc_width),
    # to_Vector( name='l4', n_filer=256, width=fc_width, offset="(11.1,0,0)", caption="FC2", height=fc_height, depth=fc_width),
    to_connection( "l3", "l4" ),

    ###### action branch ######
    to_Vector( name='l51', n_filer=4096, width=fc_width, offset="(13.1,5,0)", caption="Action-FC", height=fc_height/2, depth=fc_width),
    to_manhattan_connection( "l4", "l51" ),

    ###### value branch ######
    to_Vector( name='l52', n_filer=4096, width=fc_width, offset="(13.1,-5,0)", caption="Value-FC", height=fc_height/2, depth=fc_width),
    to_manhattan_connection( "l4", "l52" ),

    # to_Vector( name='feature', n_filer=256, width=fc_width, offset="(11.1,0,0)", caption="feature", height=fc_height, depth=fc_width),
    # to_connection( "l4", "feature" ),  


    # to_Onehot( name='feature', n_filer=9, s_filer=16, offset="(11.9,0,0)", width=2.5, height=width, depth=width),
    # to_input( 'test_img_o.png', to='(12.4,0,0)', width=im_w, height=im_w),
    to_end() 
    ]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
    
