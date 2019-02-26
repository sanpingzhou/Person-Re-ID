#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void P2SLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);

  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int num_negative = this->layer_param_.p2s_loss_param().num_negative();
  int number = num / 2;                                                        // number of images in each channel
  int number_id = number / (1 + num_negative);
  feature_probe_.Reshape(number, channels, 1, 1);
  label_probe_.Reshape(number, 1, 1, 1);
  index_probe_.Reshape(number, 1, 1, 1);
  feature_gallery_.Reshape(number, channels, 1, 1);
  label_gallery_.Reshape(number, 1, 1, 1);
  index_gallery_.Reshape(number, 1, 1, 1);
  diff_ab_.Reshape(number, channels, 1, 1);
  dist_ab_.Reshape(number, 1, 1, 1);

  temp_feature_probe_.Reshape(number_id, channels, 1, 1);                      // number of positive pairs
  temp_feature_gallery_.Reshape(number_id, channels, 1, 1);
  temp_sign_.Reshape(number_id, 1, 1, 1);
  temp_dist_ab_.Reshape(number_id, 1, 1, 1);

  feature_anchor_.Reshape(number, channels, 1, 1);
  label_anchor_.Reshape(number, 1, 1, 1);
  index_anchor_.Reshape(number, 1, 1, 1);
  feature_positive_.Reshape(number, channels, 1, 1);
  label_positive_.Reshape(number, 1, 1, 1);
  index_positive_.Reshape(number, 1, 1, 1);
  feature_negative_.Reshape(number, channels, 1, 1);
  label_negative_.Reshape(number, 1, 1, 1);
  index_negative_.Reshape(number, 1, 1, 1);

  diff_anchor_positive_.Reshape(number, channels, 1, 1);
  diff_anchor_negative_.Reshape(number, channels, 1, 1);
  diff_positive_negative_.Reshape(number, channels, 1, 1);
  dist_anchor_positive_.Reshape(number, 1, 1, 1);
  dist_anchor_negative_.Reshape(number, 1, 1, 1);
  dist_positive_negative_.Reshape(number, 1, 1, 1);

  diff_a_.Reshape(number, channels, 1, 1);
  diff_p_.Reshape(number, channels, 1, 1);
  diff_n_.Reshape(number, channels, 1, 1); 
}

template <typename Dtype>
void P2SLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
 Dtype margin1 = this->layer_param_.p2s_loss_param().margin1();
 Dtype margin2 = this->layer_param_.p2s_loss_param().margin2();
 Dtype margin3 = this->layer_param_.p2s_loss_param().margin3();
 Dtype weight = this->layer_param_.p2s_loss_param().weight();
 int num_positive = this->layer_param_.p2s_loss_param().num_positive();
 int num_negative = this->layer_param_.p2s_loss_param().num_negative();
 int choose_positive = this->layer_param_.p2s_loss_param().choose_positive();
 Dtype alpha = this->layer_param_.p2s_loss_param().alpha();
 Dtype beta = this->layer_param_.p2s_loss_param().beta();
 Dtype eta = this->layer_param_.p2s_loss_param().eta();

 int num = bottom[0]->num();
 int channels = bottom[0]->channels();
 int number = num / 2;
 int number_id = number / (1 + num_negative);

 // copy the feature representation and index to probe and gallery
 int indexProbe = 0;
 int indexGallery = 0;
 for (int i = 0; i < num; ++i) {
    if (i % 2 == 0 ) {
      caffe_copy(channels,
                 bottom[0]->cpu_data() + (i * channels),
                 feature_probe_.mutable_cpu_data() + (indexProbe * channels));
      label_probe_.mutable_cpu_data()[indexProbe] = bottom[1]->cpu_data()[i];
      index_probe_.mutable_cpu_data()[indexProbe] = i;
      ++indexProbe;
    } 
    else {
      caffe_copy(channels, bottom[0]->cpu_data() + (i * channels),
                 feature_gallery_.mutable_cpu_data() + (indexGallery * channels));
      label_gallery_.mutable_cpu_data()[indexGallery] = bottom[1]->cpu_data()[i];
      index_gallery_.mutable_cpu_data()[indexGallery] = i;
      ++indexGallery;
    }    
 }
 
 // siamese part 
 // computting the distace between probe and gallery
 caffe_sub(number * channels,
      	   feature_probe_.cpu_data(),            // input probe features
      	   feature_gallery_.cpu_data(),          // input gallery featurs
      	   diff_ab_.mutable_cpu_data());

 Dtype loss1(0.0);
 for (int i = 0; i < number; ++i) {
   dist_ab_.mutable_cpu_data()[i] = caffe_cpu_dot (channels,
                                                   diff_ab_.cpu_data() + (i * channels),
                                                   diff_ab_.cpu_data() + (i * channels));
   if (static_cast<int>(label_probe_.cpu_data()[i] == label_gallery_.cpu_data()[i])) {  // similar pairs
      Dtype cdist = std::max(dist_ab_.cpu_data()[i] - margin1, Dtype(0.0));
      if (cdist == Dtype(0.0)) {
         caffe_set(channels, Dtype(0.0), diff_ab_.mutable_cpu_data() + (i * channels));
      }
      loss1 += cdist;
    }
   else {  // dissimilar pairs
     Dtype mdist = std::max(margin2 - dist_ab_.cpu_data()[i], Dtype(0.0));
     if (mdist == Dtype(0.0)) {
        caffe_set(channels, Dtype(0.0), diff_ab_.mutable_cpu_data() + (i * channels));
     }
     loss1 += mdist;
   } 
 }

 // triplet part
 Dtype loss2(0.0);
 caffe_set(number * channels, Dtype(0.0), feature_anchor_.mutable_cpu_data());
 caffe_set(number * channels, Dtype(0.0), feature_positive_.mutable_cpu_data());
 caffe_set(number * channels, Dtype(0.0), feature_negative_.mutable_cpu_data());

 // create set according to positive
 int countPositive = 0;
 for (int i = 0; i < number; i = i + (1 + num_negative)) { // similar pairs
    if (label_probe_.cpu_data()[i] == label_gallery_.cpu_data()[i]) {
       caffe_copy(channels,
                  feature_probe_.cpu_data() + (i * channels),
                  temp_feature_probe_.mutable_cpu_data() + (countPositive * channels));
       caffe_copy(channels, 
                  feature_gallery_.cpu_data() + (i * channels),
                  temp_feature_gallery_.mutable_cpu_data() + (countPositive * channels));
       temp_dist_ab_.mutable_cpu_data()[countPositive] = dist_ab_.cpu_data()[i];
       temp_sign_.mutable_cpu_data()[countPositive] = i;
       ++ countPositive;
    }
 }
 
 for (int i = 0; i < number_id; i = i + num_positive) {  // rank positive in local (large to small)
    for (int j = i; j < num_positive; ++j) {
       if (temp_dist_ab_.cpu_data()[i] < temp_dist_ab_.cpu_data()[j]) {
          Dtype tempDist = temp_dist_ab_.cpu_data()[i];
          temp_dist_ab_.mutable_cpu_data()[i] = temp_dist_ab_.cpu_data()[j];
          temp_dist_ab_.mutable_cpu_data()[j] = tempDist;

          int tempSign = temp_sign_.cpu_data()[i];
          temp_sign_.mutable_cpu_data()[i] = temp_sign_.cpu_data()[j];
          temp_sign_.mutable_cpu_data()[j] = tempSign;
       }      
    }
 }

 for (int i = choose_positive; i < number_id; i = i + num_positive) {
    if (num_positive > choose_positive) {
      int tempindex = temp_sign_.cpu_data()[i];
      caffe_set(channels, Dtype(0.0), feature_probe_.mutable_cpu_data() + (tempindex * channels));
      caffe_set(channels, Dtype(0.0), feature_gallery_.mutable_cpu_data() + (tempindex * channels));
      label_probe_.mutable_cpu_data()[tempindex] = -1;
      label_gallery_.mutable_cpu_data()[tempindex] = -1;
    } else {
       break;
    }

 }

 // generate triplets
 for (int i = 0; i < number; ++i) { 
    if (label_probe_.cpu_data()[i] == label_gallery_.cpu_data()[i] && label_probe_.cpu_data()[i] != -1 && label_gallery_.cpu_data()[i] != -1 ) {  // similar pairs
       int indexSemiHardNegative;
       int countSemiHardNegative = 0;
       while (1) {
	 indexSemiHardNegative = rand() % (num/2);
         if (label_probe_.cpu_data()[i] != label_gallery_.cpu_data()[indexSemiHardNegative]) {
             Dtype relativeDistance = std::max(dist_ab_.cpu_data()[indexSemiHardNegative] - dist_ab_.cpu_data()[i], Dtype(0.0));
             ++countSemiHardNegative;
             if (relativeDistance == Dtype(0.0) || label_gallery_.cpu_data()[indexSemiHardNegative] == -1 || countSemiHardNegative == 10000) {
             	break;
             }
         }
       }
       caffe_copy(channels, feature_probe_.cpu_data() + (i * channels),
                  feature_anchor_.mutable_cpu_data() + (i * channels));
       caffe_copy(channels, feature_gallery_.cpu_data() + (i * channels),
                  feature_positive_.mutable_cpu_data() + (i * channels));
       caffe_copy(channels, feature_gallery_.cpu_data() + (indexSemiHardNegative * channels),
                  feature_negative_.mutable_cpu_data() + (i * channels));
       label_anchor_.mutable_cpu_data()[i] = label_probe_.cpu_data()[i];
       label_positive_.mutable_cpu_data()[i] = label_gallery_.cpu_data()[i];
       label_negative_.mutable_cpu_data()[i] = label_gallery_.cpu_data()[indexSemiHardNegative];
       index_anchor_.mutable_cpu_data()[i] = index_probe_.cpu_data()[i];
       index_positive_.mutable_cpu_data()[i] = index_gallery_.cpu_data()[i];
       index_negative_.mutable_cpu_data()[i] = index_gallery_.cpu_data()[indexSemiHardNegative];
   
    } 
 }
 // initialization to back-propagation
 caffe_set(number * channels, Dtype(0.0), diff_a_.mutable_cpu_data());
 caffe_set(number * channels, Dtype(0.0), diff_p_.mutable_cpu_data());
 caffe_set(number * channels, Dtype(0.0), diff_n_.mutable_cpu_data());

 caffe_sub(number * channels, 
           feature_anchor_.cpu_data(),                // anchor
	   feature_positive_.cpu_data(),              // positive
           diff_anchor_positive_.mutable_cpu_data()); // anchor - positive
 caffe_sub(number * channels, 
           feature_anchor_.cpu_data(),                // anchor
	   feature_negative_.cpu_data(),              // negative
           diff_anchor_negative_.mutable_cpu_data()); // anchor - negative
 caffe_sub(number * channels, 
           feature_positive_.cpu_data(),                // positive
	   feature_negative_.cpu_data(),                // negative
           diff_positive_negative_.mutable_cpu_data()); // positive - negative
 
 for (int i = 0; i < number; ++i) {
    dist_anchor_positive_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
                                                                diff_anchor_positive_.cpu_data() + (i * channels),
                                                                diff_anchor_positive_.cpu_data() + (i * channels));
    dist_anchor_negative_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
                                                                diff_anchor_negative_.cpu_data() + (i * channels),
                                                                diff_anchor_negative_.cpu_data() + (i * channels));
    dist_positive_negative_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
                                                                  diff_positive_negative_.cpu_data() + (i * channels),
                                                                  diff_positive_negative_.cpu_data() + (i * channels));
    Dtype temp_beta = beta - eta * (dist_anchor_negative_.cpu_data()[i] - dist_positive_negative_.cpu_data()[i]);
    Dtype mu = alpha + temp_beta;
    Dtype nu = alpha - temp_beta;
    caffe_cpu_axpby(channels,  
                    Dtype(1.0),  
                    diff_anchor_positive_.cpu_data() + (i*channels),  
                    Dtype(1.0),  
                    diff_a_.mutable_cpu_data() + (i*channels));
    caffe_cpu_axpby(channels,  
                    Dtype(-1.0)*mu,  
                    diff_anchor_negative_.cpu_data() + (i*channels),  
                    Dtype(1.0),  
                    diff_a_.mutable_cpu_data() + (i*channels));       // anchor
    caffe_cpu_axpby(channels,  
                    Dtype(-1.0),  
                    diff_anchor_positive_.cpu_data() + (i*channels),  
                    Dtype(1.0),  
                    diff_p_.mutable_cpu_data() + (i*channels));
    caffe_cpu_axpby(channels,  
                    Dtype(-1.0)*nu,  
                    diff_positive_negative_.cpu_data() + (i*channels),  
                    Dtype(1.0),  
                    diff_p_.mutable_cpu_data() + (i*channels));      // positive
   caffe_cpu_axpby(channels,  
                   mu,  
                   diff_anchor_negative_.cpu_data() + (i*channels),  
                   Dtype(1.0),  
                   diff_n_.mutable_cpu_data() + (i*channels));
   caffe_cpu_axpby(channels,  
                   nu,  
                   diff_positive_negative_.cpu_data() + (i*channels),  
                   Dtype(1.0),  
                   diff_n_.mutable_cpu_data() + (i*channels));     // back-propagation for negative
    Dtype sdist =  std::max(margin3 + dist_anchor_positive_.cpu_data()[i] - mu*dist_anchor_negative_.cpu_data()[i] - nu*dist_positive_negative_.cpu_data()[i], Dtype(0.0));
    if (sdist == Dtype(0.0)) {
      caffe_set(channels, Dtype(0.0), diff_a_.mutable_cpu_data() + (i * channels));
      caffe_set(channels, Dtype(0.0), diff_p_.mutable_cpu_data() + (i * channels));
      caffe_set(channels, Dtype(0.0), diff_n_.mutable_cpu_data() + (i * channels));
    }
    loss2 += sdist;
 }

 Dtype loss = loss1 + weight * loss2;
 loss = loss / static_cast<Dtype>(bottom[0]->num());
 top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void P2SLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype weight = this->layer_param_.p2s_loss_param().weight();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  Dtype* bout = bottom[0] ->mutable_cpu_diff();
  int number = num / 2;
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) { 
  const Dtype alpha = top[0]->cpu_diff()[0] / static_cast<Dtype>(number);
  for (int i = 0; i < number; ++i) {
      // back-propagation for probe and gallery 
      // siamese part
      int tempindex_probe = index_probe_.cpu_data()[i];
      int tempindex_gallery = index_gallery_.cpu_data()[i];
      int tempindex_anchor = index_anchor_.cpu_data()[i];
      int tempindex_positive = index_positive_.cpu_data()[i];
      int tempindex_negative = index_negative_.cpu_data()[i];
      if (label_probe_.cpu_data()[i] == label_gallery_.cpu_data()[i]) { // similar pairs
        caffe_cpu_axpby(channels,  // for probe
                        alpha * Dtype(1.0), 
                        diff_ab_.cpu_data() + (i * channels),
                        Dtype(0.0),
                        bout + (tempindex_probe * channels));

        caffe_cpu_axpby(channels, // for gallery 
                        alpha * Dtype(-1.0), 
                        diff_ab_.cpu_data() + (i * channels),
                        Dtype(0.0),
                        bout + (tempindex_gallery * channels));
      }
      else { // dissimialr pairs
        caffe_cpu_axpby(channels, // for probe
                        alpha * Dtype(-1.0),
                        diff_ab_.cpu_data() + (i * channels),
                        Dtype(0.0),
                        bout + (tempindex_probe * channels));
        caffe_cpu_axpby(channels, // for gallery
                        alpha * Dtype(1.0),
                        diff_ab_.cpu_data() + (i * channels),
                        Dtype(0.0),
                        bout + (tempindex_gallery * channels));
      }
      // triplet part
      caffe_cpu_axpby(channels,  // anchor
		      alpha * weight,
		      diff_a_.cpu_data() + (i * channels),
		      Dtype(1.0),
		      bout + (tempindex_anchor * channels));
      caffe_cpu_axpby(channels, // positive
		      alpha * weight,
		      diff_p_.cpu_data() + (i * channels),
		      Dtype(1.0),
		      bout + (tempindex_positive * channels));
      caffe_cpu_axpby(channels, // negative
		      alpha * weight,
		      diff_n_.cpu_data() + (i * channels),
		      Dtype(1.0),
		      bout + (tempindex_negative * channels));
  }
 }
}

#ifdef CPU_ONLY
STUB_GPU(P2SLossLayer);
#endif

INSTANTIATE_CLASS(P2SLossLayer);
REGISTER_LAYER_CLASS(P2SLoss);

}  // namespace caffe
