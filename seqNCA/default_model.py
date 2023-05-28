
import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *

in_width = 30
im_w = in_width / 5


out_width = 10
im_out_w = out_width / 5

fc_height = 40
fc_width = 1

arch = [ 
    to_head('..'), 
    to_cor(),
    to_begin(),

    ###### feature ######
    to_Onehot( name='input', n_filer=3, s_filer=32, offset="(-3.5,0,0)", width=2.5, height=in_width, depth=in_width),
    to_input( 'test_img.png', to="(-3,1.2,0.6)", name='input', width=im_w/2, height=im_w/2),
    to_ConvRelu( name='l1', s_filer=32, n_filer=64, width=16, caption="Conv2D-7$\\times$7--ReLu", height=in_width, depth=in_width, offset="(-0.5,0,0)"),
    to_connection( "input", "l1" ),
    to_ConvRelu( name='l2', s_filer=16, n_filer=64, width=16, offset="(4.7,0,0)", caption="Conv2D-7$\\times$7--ReLu", height=in_width/2, depth=in_width/2),
    to_connection( "l1", "l2" ),
    # to_Vector( name='l3', n_filer=4096, width=fc_width, offset="(9.1,0,0)", caption="FC1", height=fc_height*2, depth=fc_width),
    to_Conv1DRelu( name='l3', s_filer=1, n_filer=256, offset="(9.1,0,0)", caption="FC1--ReLu", width=2, height=fc_height*1.5, depth=fc_width),
    to_connection( "l2", "l3" ),
    to_Conv1DRelu( name='l4', s_filer=1, n_filer=64, offset="(11.1,0,0)", caption="FC2--ReLu", width=2, height=fc_height, depth=fc_width),
    # to_Vector( name='l4', n_filer=256, width=fc_width, offset="(11.1,0,0)", caption="FC2", height=fc_height, depth=fc_width),
    to_connection( "l3", "l4" ),

    ###### action branch ######
    to_Conv1DRelu( name='l51', n_filer=8, width=fc_width, offset="(13.6,2.5,0)", caption="ActionFC--ReLu", height=fc_height/4, depth=fc_width),
    # to_Conv1DRelu_nolabel( name='l51', n_filer=8, width=fc_width, offset="(13.1,2.5,0)", caption="Action-FC", height=fc_height/4, depth=fc_width),
    to_manhattan_connection( "l4", "l51", pos=1 ),

    to_Onehot( name='output_action', n_filer=3, s_filer=2, offset="(16,2.5,0)", width=2.5, height=out_width, depth=out_width),
    to_input( 'test_img_o.png', to="(16.5,2.5,0)", name='output_action', width=im_out_w, height=im_out_w),
    to_connection( "l51", "output_action" ),


    ###### value branch ######
    to_Conv1DRelu( name='l52', n_filer=1, width=fc_width, offset="(13.6,-2.5,0)", caption="ValueFC--ReLu", height=1, depth=fc_width),
    to_manhattan_connection( "l4", "l52", pos=1 ),

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
    
