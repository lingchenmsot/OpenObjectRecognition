#pragma once
#include "training.h"
#include "svm_detector.h"


void exec_train_svm(const string & pos_dir, const string & pos, const string & neg_dir, const string & neg);
void exec_detect_pic(const string & svm_path);
void get_hard_examples(const string & output_dir, const string & svm_path, const string & dir,
	const string & names);
