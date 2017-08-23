#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/focal_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FocalLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
  
  alpha_ = this->layer_param_.focal_loss_param().alpha();
  gamma_ = this->layer_param_.focal_loss_param().gamma();
  CHECK_GT(alpha_, 0) << "alpha must be greater than zero.";
  CHECK_GE(gamma_, 0) << "gamma must be equal or greater than zero.";
}

template <typename Dtype>
void FocalLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
  
  // log(prob)
  log_prob_.ReshapeLike(*bottom[0]);
  // all elements is set to 1, in order to facilitate the calculation
  aux_ones_.ReshapeLike(*bottom[0]);
  caffe_set(bottom[0]->count(), Dtype(1), aux_ones_.mutable_cpu_data());
  // the modulating factor of the cross entropy loss
  // alpha * (1 - prob) ^ gamma
  modulating_factor_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
Dtype FocalLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void FocalLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;
  int count = 0;
  Dtype loss = 0;
  
  int total_num = prob_.count();
  const Dtype* aux_ones_data = aux_ones_.cpu_data();
  const Dtype* log_prob_data = log_prob_.cpu_data();
  const Dtype* mfactor_data = modulating_factor_.cpu_data();
  Dtype* log_prob_mutable_data = log_prob_.mutable_cpu_data();
  Dtype* mfactor_mutable_data = modulating_factor_.mutable_cpu_data();

  // log(p)
  caffe_log(total_num, prob_data, log_prob_mutable_data);
  // 1 - p
  caffe_sub(total_num, aux_ones_data, prob_data, mfactor_mutable_data);
  // (1 - p) ^ gamma
  caffe_powx(total_num, mfactor_data, gamma_, mfactor_mutable_data);
  // alpha * (1 - p) ^ gamma
  caffe_scal(total_num, alpha_, mfactor_mutable_data);
  
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));
      // cross-entropy loss
      // loss = -log(probability corresponding to the correct class)
      //loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
      //                     Dtype(FLT_MIN)));
      int index = i * dim + label_value * inner_num_ + j;
      // focal loss
      loss -= mfactor_data[index] * log_prob_data[index];
      ++count;
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void FocalLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int dim = prob_.count() / outer_num_;
    int count = 0;

    const Dtype* log_prob_data = log_prob_.cpu_data();
    const Dtype* mfactor_data = modulating_factor_.cpu_data();
    
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        } else {
          // cross-entropy loss BP
          //bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
          int index = i * dim + label_value * inner_num_ + j;
          Dtype fl_part_gradient = mfactor_data[index] - gamma_ * prob_data[index] * 
                                   log_prob_data[index] * mfactor_data[index] / 
                                   std::max(1-prob_data[index], Dtype(FLT_MIN));
          
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            int index_detail = i * dim + c * inner_num_ + j;
            if (label_value == c) { // corresponding class
              bottom_diff[index_detail] = fl_part_gradient * (prob_data[index] - 1);
            }
            else {
              bottom_diff[index_detail] = fl_part_gradient * prob_data[index_detail];
            }
          }
          ++count;
        }
      }
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] /
                        get_normalizer(normalization_, count);
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(FocalLossLayer);
#endif

INSTANTIATE_CLASS(FocalLossLayer);
REGISTER_LAYER_CLASS(FocalLoss);

}  // namespace caffe
