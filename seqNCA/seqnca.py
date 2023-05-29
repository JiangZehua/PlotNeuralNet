
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

# used for adjusting the thickness of the convolutional layer
feat_width = 10

# used for moving the input image around
im_x = -0.7
im_y = 0.85

arch = [ 
    to_head('..'), 
    to_cor(),
    to_begin(),

    ###### feature ######
    to_Onehot( name='input', n_filer=3, s_filer=32, offset="(-3.5,0,0)", width=2.5, height=in_width, depth=in_width),
    to_input( 'im-226_.png', to="(-3,"+str(im_y)+","+str(im_x)+")", name='input', width=im_w/2, height=im_w/2),
    to_ConvRelu( name='l1', s_filer=32, n_filer=64, width=feat_width, caption="Conv2D-3$\\times$3--ReLu", height=in_width, depth=in_width, offset="(-0.5,0,0)"),
    to_connection( "input", "l1" ),
    to_Onehot( name='cropped_feat', n_filer=64, s_filer=3, offset="(-0.5,"+str(im_y)+","+str(im_x)+")", width=feat_width, height=in_width/5, depth=in_width/5),

    ###### action branch ######
    to_FcRelu( name='l21', n_filer=64, width=fc_width, offset="(6.1,3,0)", caption="ActionFC1--ReLu", depth=fc_height/1.5, height=fc_width),
    to_manhattan_connection( "cropped_feat", "l21", pos=2 ),
    to_FcRelu( name='l31', n_filer=8, width=fc_width, offset="(8.1,3,0)", caption="ActionFC2--ReLu", depth=fc_height/4, height=fc_width),
    to_connection( "l21", "l31" ),
    # to_manhattan_connection( "l3", "l41", pos=1 ),

    to_Onehot( name='output_action', n_filer=3, s_filer=2, offset="(10,3,0)", width=2.5, height=out_width, depth=out_width),
    to_input( 'im-227.png', to="(10.5,3,0)", name='output_action', width=im_out_w, height=im_out_w),
    to_connection( "l31", "output_action" ),


    ###### value branch ######
    to_FcRelu( name='l22', n_filer=64, width=fc_width, offset="(6.1,-3,0)", caption="ValueFC1--ReLu", depth=fc_height/1.5,height=fc_width),
    to_manhattan_connection( "l1", "l22", pos=2 ),
    to_FcRelu( name='l32', n_filer=1, width=fc_width, offset="(8.1,-3,0)", caption="ValueFC2--ReLu", depth=fc_width, height=fc_width),
    to_connection( "l22", "l32" ),
    # to_manhattan_connection( "l3", "l42", pos=1 ),

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
    
