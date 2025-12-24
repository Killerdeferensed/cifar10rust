use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig, Conv3d,Conv3dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig,},
        Dropout, DropoutConfig, Linear, LinearConfig, Relu, LeakyRelu, LeakyReluConfig
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Model<B: Backend>{
    /*First */
    // conv1: Conv3d<B>,
    // conv2: Conv2d<B>,
    // pool: AdaptiveAvgPool2d,
    // dropout: Dropout,
    // linear1: Linear<B>,
    // linear2: Linear<B>,
    // activation: Relu,

    /*Second */
    // conv_prec_1:Conv3d<B>,
    // conv_prec_2:Conv2d<B>,
    // conv_prec_3:Conv2d<B>,
    // pool_prec: AdaptiveAvgPool2d,
    // linear_prec: Linear<B>,
    // dropout_prec: Dropout,
    // activation_prec: LeakyRelu,

    // conv_bulk_1:Conv3d<B>,
    // conv_bulk_2:Conv2d<B>,
    // dropout_bulk: Dropout,
    // linear_bulk: Linear<B>,
    // pool_bulk: AdaptiveAvgPool2d,
    // activation_bulk: Relu,

    // linear_global: Linear<B>

    /*Third */
    conv1: Conv3d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    pool1: MaxPool2d,
    conv4: Conv2d<B>,
    conv5: Conv2d<B>,
    conv6: Conv2d<B>,
    pool2: MaxPool2d,
    conv7: Conv2d<B>,
    conv8: Conv2d<B>,
    conv9: Conv2d<B>,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
    dropout: Dropout,
}

impl<B: Backend> Model<B>  {
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2>{

        let [batch_size, heigth, width, depth] = images.dims();
        let x = images.reshape([batch_size, 1, heigth, width, depth]);

        /*First */
        // let x = self.conv1.forward(x); //[bs,8,30,30,1]
        // let x: Tensor<B, 4> = x.squeeze(4);
        // let x = self.dropout.forward(x);
        // let x = self.conv2.forward(x); //[bs,16,28,28]
        // let x = self.dropout.forward(x);
        // let x = self.activation.forward(x); 

        // let x = self.pool.forward(x); //[bs,16,7,7]
        // let x = x.reshape([batch_size, 16*7*7]);
        // let x = self.linear1.forward(x);
        // let x = self.dropout.forward(x);
        // let x = self.activation.forward(x);

        // self.linear2.forward(x)
        // let x:Tensor<B, 4> = x.squeeze(4);


        /*Second */
        // let prec = images.clone().reshape([batch_size, 1, heigth, width, depth]);
        // let bulk = images.reshape([batch_size, 1, heigth, width, depth]);

        // let prec = self.conv_prec_1.forward(prec);
        // let prec: Tensor<B, 4> = prec.squeeze(4);
        // let prec = self.dropout_prec.forward(prec);
        // let prec = self.conv_prec_2.forward(prec);
        // let prec = self.dropout_prec.forward(prec);
        // let prec = self.conv_prec_3.forward(prec);
        // let prec = self.dropout_prec.forward(prec);
        // let prec = self.activation_prec.forward(prec);
        // let prec = self.pool_prec.forward(prec);
        // let prec = prec.reshape([batch_size, 36*9*9]);
        // let prec = self.linear_prec.forward(prec);
        // let prec = self.dropout_prec.forward(prec);
        // let prec = self.activation_prec.forward(prec);

        // let bulk = self.conv_bulk_1.forward(bulk);
        // let bulk: Tensor<B, 4> = bulk.squeeze(4);
        // let bulk = self.dropout_bulk.forward(bulk);
        // let bulk = self.conv_bulk_2.forward(bulk);
        // let bulk = self.dropout_bulk.forward(bulk);
        // let bulk = self.activation_bulk.forward(bulk);
        // let bulk = self.pool_bulk.forward(bulk);
        // let bulk = bulk.reshape([batch_size, 16*7*7]);
        // let bulk = self.linear_bulk.forward(bulk);
        // let bulk = self.dropout_bulk.forward(bulk);
        // let bulk = self.activation_bulk.forward(bulk);

        // let res = (prec+bulk)/2;

        // self.linear_global.forward(res)


        /*Third */
        let x = self.conv1.forward(x);
        let x: Tensor<B, 4> = x.squeeze(4);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);

        let x = self.conv2.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);

        let x = self.conv3.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);

        let x = self.pool1.forward(x);

        let x = self.conv4.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);

        let x = self.conv5.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);

        let x = self.conv6.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);

        let x = self.pool2.forward(x);

        let x = self.conv7.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);

        let x = self.conv8.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);

        let x = self.conv9.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);

        let x = x.reshape([batch_size, 266*5*5]);
        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);

        self.linear2.forward(x)
    }
}

#[derive(Config,Debug)]
pub struct ModelConfig {
    num_classes:usize,
    hiden_size:usize,
    #[config(default = "0.5")]
    dropout: f64,
    // #[config(default = "0.5")]
    // dropour_bulk: f64
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B>{
        Model{ 
            /* First try*/
            // conv1: Conv3dConfig::new([1, 8], [3,3,3]).init(device),
            // conv2: Conv2dConfig::new([8, 16], [3,3]).init(device),
            // pool: AdaptiveAvgPool2dConfig::new([7, 7]).init(),
            // activation: Relu::new(),
            // linear1: LinearConfig::new(16 * 7 * 7, self.hiden_size).init(device),
            // linear2: LinearConfig::new(self.hiden_size, self.num_classes).init(device),
            // dropout: DropoutConfig::new(self.dropout).init()

            /*Second try */
            // conv_prec_1: Conv3dConfig::new([1,12], [1,1,3]).init(device),
            // conv_prec_2: Conv2dConfig::new([12,24], [2,2]).init(device),
            // conv_prec_3:  Conv2dConfig::new([24,36], [2,2]).init(device),
            // pool_prec:  AdaptiveAvgPool2dConfig::new([9,9]).init(),
            // linear_prec: LinearConfig::new(36*9*9, self.hiden_size).init(device),
            // dropout_prec: DropoutConfig::new(self.dropout_prec).init(),
            // activation_prec: LeakyReluConfig::new().init(),

            // conv_bulk_1: Conv3dConfig::new([1,8], [3,3,3]).init(device),
            // conv_bulk_2: Conv2dConfig::new([8,16], [3,3]).init(device),
            // dropout_bulk: DropoutConfig::new(self.dropour_bulk).init(),
            // pool_bulk: AdaptiveAvgPool2dConfig::new([7,7]).init(),
            // linear_bulk: LinearConfig::new(16*7*7, self.hiden_size).init(device),
            // activation_bulk: Relu::new(),

            // linear_global: LinearConfig::new(self.hiden_size, self.num_classes).init(device),

            /*Third try */
            conv1: Conv3dConfig::new([1,32], [3,3,3]).with_padding(nn::PaddingConfig3d::Explicit(1, 1, 0)).init(device),
            conv2: Conv2dConfig::new([32,64], [3,3]).with_padding(nn::PaddingConfig2d::Same).init(device),
            conv3: Conv2dConfig::new([64,96], [3,3]).with_padding(nn::PaddingConfig2d::Same).init(device),
            pool1: MaxPool2dConfig::new([4,4]).with_strides([2,2]).init(),
            conv4: Conv2dConfig::new([96, 128], [3,3]).with_padding(nn::PaddingConfig2d::Same).init(device),
            conv5: Conv2dConfig::new([128,160], [3,3]).with_padding(nn::PaddingConfig2d::Same).init(device),
            conv6: Conv2dConfig::new([160,192], [3,3]).init(device),
            pool2: MaxPool2dConfig::new([4,4]).with_strides([2,2]).init(),
            conv7: Conv2dConfig::new([192,224], [3,3]).with_padding(nn::PaddingConfig2d::Same).init(device),
            conv8: Conv2dConfig::new([224,256], [3,3]).with_padding(nn::PaddingConfig2d::Same).init(device),
            conv9: Conv2dConfig::new([256,266], [3,3]).with_padding(nn::PaddingConfig2d::Same).init(device),
            linear1: LinearConfig::new(266*5*5, self.hiden_size).init(device),
            linear2: LinearConfig::new(self.hiden_size, self.num_classes).init(device),
            activation: Relu::new(),
            dropout: DropoutConfig::new(self.dropout).init(),

        }
    }
}