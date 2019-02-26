#include <algorithm>  
#include <vector>  
  
#include "caffe/layer.hpp"  
#include "caffe/loss_layers.hpp"  
#include "caffe/util/io.hpp"  
#include "caffe/util/math_functions.hpp"  
  
namespace caffe {  
  
template <typename Dtype>  
void SymmetricTripletLossLayer<Dtype>::LayerSetUp(  
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {  
  LossLayer<Dtype>::LayerSetUp(bottom, top);  
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());  
  CHECK_EQ(bottom[1]->num(), bottom[2]->num());  
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());  
  CHECK_EQ(bottom[1]->channels(), bottom[2]->channels());  
  CHECK_EQ(bottom[0]->height(), 1);  
  CHECK_EQ(bottom[0]->width(), 1);  
  CHECK_EQ(bottom[1]->height(), 1);  
  CHECK_EQ(bottom[1]->width(), 1);  
  CHECK_EQ(bottom[2]->height(), 1);  
  CHECK_EQ(bottom[2]->width(), 1);  

  diff_ap_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);  
  diff_an_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1); 
  diff_pn_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);

  diff_a_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1); 
  diff_p_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1); 
  diff_n_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1); 
   
  dist_ap_.Reshape(bottom[0]->num(), 1, 1, 1);
  dist_an_.Reshape(bottom[0]->num(), 1, 1, 1);
  dist_pn_.Reshape(bottom[0]->num(), 1, 1, 1);
}  
  
template <typename Dtype>  
void SymmetricTripletLossLayer<Dtype>::Forward_cpu(  
    const vector<Blob<Dtype>*>& bottom,  
    const vector<Blob<Dtype>*>& top) {
  Dtype margin = this->layer_param_.symmetric_triplet_loss_param().margin();
  Dtype alpha = this->layer_param_.symmetric_triplet_loss_param().alpha();
  Dtype beta = this->layer_param_.symmetric_triplet_loss_param().beta();
  Dtype eta = this->layer_param_.symmetric_triplet_loss_param().eta();
  int channels = bottom[0]->channels();    
  int count = bottom[0]->count();
  caffe_sub(count,  
            bottom[0]->cpu_data(),          // a  
            bottom[1]->cpu_data(),          // p  
            diff_ap_.mutable_cpu_data());   // a_i-p_i  
  caffe_sub(count,  
            bottom[0]->cpu_data(),         // a  
            bottom[2]->cpu_data(),         // n  
            diff_an_.mutable_cpu_data());  // a_i-n_i  
  caffe_sub(count,  
            bottom[1]->cpu_data(),         // p  
            bottom[2]->cpu_data(),         // n  
            diff_pn_.mutable_cpu_data());  // p_i-n_i  

  Dtype loss(0.0);
  caffe_set(count, Dtype(0.0), diff_a_.mutable_cpu_data());
  caffe_set(count, Dtype(0.0), diff_p_.mutable_cpu_data());
  caffe_set(count, Dtype(0.0), diff_n_.mutable_cpu_data());
  for (int i = 0; i < bottom[0]->num(); ++i) {  
    dist_ap_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,  
                                                   diff_ap_.cpu_data() + (i*channels),
                                                   diff_ap_.cpu_data() + (i*channels));  
    dist_an_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,  
                                                   diff_an_.cpu_data() + (i*channels),
                                                   diff_an_.cpu_data() + (i*channels));
    dist_pn_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,  
                                                   diff_pn_.cpu_data() + (i*channels),
                                                   diff_pn_.cpu_data() + (i*channels));
    Dtype temp = beta - eta * (dist_an_.cpu_data()[i] - dist_pn_.cpu_data()[i]);
    Dtype mu = alpha + temp;
    Dtype nu = alpha - temp;
    caffe_cpu_axpby(channels,  
                    Dtype(1.0),  
                    diff_ap_.cpu_data() + (i*channels),  
                    Dtype(1.0),  
                    diff_a_.mutable_cpu_data() + (i*channels));
    caffe_cpu_axpby(channels,  
                    Dtype(-1.0)*mu,  
                    diff_an_.cpu_data() + (i*channels),  
                    Dtype(1.0),  
                    diff_a_.mutable_cpu_data() + (i*channels));       // anchor
    caffe_cpu_axpby(channels,  
                    Dtype(-1.0),  
                    diff_ap_.cpu_data() + (i*channels),  
                    Dtype(1.0),  
                    diff_p_.mutable_cpu_data() + (i*channels));
    caffe_cpu_axpby(channels,  
                    Dtype(-1.0)*nu,  
                    diff_pn_.cpu_data() + (i*channels),  
                    Dtype(1.0),  
                    diff_p_.mutable_cpu_data() + (i*channels));      // positive
   caffe_cpu_axpby(channels,  
                   mu,  
                   diff_an_.cpu_data() + (i*channels),  
                   Dtype(1.0),  
                   diff_n_.mutable_cpu_data() + (i*channels));
   caffe_cpu_axpby(channels,  
                   nu,  
                   diff_pn_.cpu_data() + (i*channels),  
                   Dtype(1.0),  
                   diff_n_.mutable_cpu_data() + (i*channels));     // back-propagation for negative
    Dtype rdist =  std::max(margin + dist_ap_.cpu_data()[i] - mu*dist_an_.cpu_data()[i] - nu*dist_pn_.cpu_data()[i], Dtype(0.0));
    if (rdist == Dtype(0.0)) {
      caffe_set(channels, Dtype(0.0), diff_a_.mutable_cpu_data() + (i * channels));
      caffe_set(channels, Dtype(0.0), diff_p_.mutable_cpu_data() + (i * channels));
      caffe_set(channels, Dtype(0.0), diff_n_.mutable_cpu_data() + (i * channels));
    }
    loss += rdist;
 }
  loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);  
  top[0]->mutable_cpu_data()[0] = loss;  
}  
  
template <typename Dtype>  
void SymmetricTripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,  
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {   
  for (int i = 0; i < 3; ++i) {  
    if (propagate_down[i]) {   
      Dtype alpha = top[0]->cpu_diff()[0]/static_cast<Dtype>(bottom[i]->num());  
      int num = bottom[i]->num();  
      int channels = bottom[i]->channels();  
      for (int j = 0; j < num; ++j) {  
        Dtype* bout = bottom[i]->mutable_cpu_diff();  
        if (i == 0) { // a  
                  caffe_cpu_axpby(channels,  
                                  alpha,  
                                  diff_a_.cpu_data() + (j*channels),  
                                  Dtype(0.0),  
                                  bout + (j*channels));  
  
        } else if (i == 1) {  // p  
                         caffe_cpu_axpby(channels,  
                                         alpha,  
                                         diff_p_.cpu_data() + (j*channels),  
                                         Dtype(0.0),  
                                         bout + (j*channels));    
        } else if (i == 2) {  // n  
                         caffe_cpu_axpby(channels,  
                                         alpha,  
                                         diff_n_.cpu_data() + (j*channels),  
                                         Dtype(0.0),  
                                         bout + (j*channels));    
        }  
      }
    } 
  }  
}  
  
#ifdef CPU_ONLY  
STUB_GPU(SymmetricTripletLossLayer);  
#endif  
  
INSTANTIATE_CLASS(SymmetricTripletLossLayer);  
REGISTER_LAYER_CLASS(SymmetricTripletLoss);  
  
}  // namespace caffe
