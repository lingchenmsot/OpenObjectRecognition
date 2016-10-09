#pragma once
#include "training.h"
#include "svm_detector.h"


void exec_train_svm(const string & pos_dir, const string & pos, const string & neg_dir, const string & neg);
void exec_detect_pic(const string & svm_path);

