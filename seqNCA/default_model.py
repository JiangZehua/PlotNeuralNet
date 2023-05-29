
import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *

in_width = 30
im_w = in_width / 5


out_width = 5
im_out_w = out_width / 5

fc_height = 40
fc_width = 1.5

conv_width = 10


arch = [ 
    to_head('..'), 
    to_cor(),
    to_begin(),

    ###### feature ######
    to_Onehot( name='input', n_filer=3, s_filer=32, offset="(-3.5,0,0)", width=2.5, height=in_width, depth=in_width),
    to_input( 'im-226_.png', to="(-3,0.85, -0.7)", name='input', width=im_w/2, height=im_w/2),
    to_ConvRelu( name='l1', s_filer=32, n_filer=64, width=conv_width, caption="Conv2D-7$\\times$7--ReLu", height=in_width, depth=in_width, offset="(-0.5,0,0)"),
    to_connection( "input", "l1" ),
    to_ConvRelu( name='l2', s_filer=16, n_filer=64, width=conv_width, offset="(3.7,0,0)", caption="Conv2D-7$\\times$7--ReLu", height=in_width/2, depth=in_width/2),
    to_connection( "l1", "l2" ),
    # to_Vector( name='l3', n_filer=4096, width=fc_width, offset="(9.1,0,0)", caption="FC1", height=fc_height*2, depth=fc_width),
    to_FcRelu( name='l3', s_filer=1, n_filer=256, offset="(8.6,0,0)", caption="FC1--ReLu", width=fc_width, depth=fc_height*1.5, height=fc_width),
    to_connection( "l2", "l3" ),
    to_FcRelu( name='l4', s_filer=1, n_filer=64, offset="(10.6,0,0)", caption="FC2--ReLu", width=fc_width, depth=fc_height, height=fc_width),
    # to_Vector( name='l4', n_filer=256, width=fc_width, offset="(11.1,0,0)", caption="FC2", height=fc_height, depth=fc_width),
    to_connection( "l3", "l4" ),

    ###### action branch ######
    to_FcRelu( name='l51', n_filer=32, width=fc_width, offset="(14,2.5,0)", caption="ActionFC--ReLu", depth=fc_height/2, height=fc_width),
    # to_FcRelu_nolabel( name='l51', n_filer=8, width=fc_width, offset="(13.1,2.5,0)", caption="Action-FC", height=fc_height/4, depth=fc_width),
    to_manhattan_connection( "l4", "l51", pos=1.2 ),

    to_Onehot( name='output_action', n_filer=3, s_filer=4, offset="(16,2.5,0)", width=2.5, height=out_width, depth=out_width),
    to_input( 'im-227.png', to="(16.5,2.5,0)", name='output_action', width=im_out_w, height=im_out_w),
    to_connection( "l51", "output_action" ),


    ###### value branch ######
    to_FcRelu( name='l52', n_filer=1, width=fc_width, offset="(13.6,-2.5,0)", caption="ValueFC--ReLu", depth=fc_width, height=fc_width),
    to_manhattan_connection( "l4", "l52", pos=1.2 ),

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
    
