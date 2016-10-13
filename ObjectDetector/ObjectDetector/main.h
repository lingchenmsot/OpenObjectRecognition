#pragma once
#include "train.h"
#include "detect.h"


void exec_train_svm(const string & pos_dir, const string & pos, const string & neg_dir, const string & neg, const Size & cell_size);
void exec_detect_pic(const string & svm_path, const Size & cell_size);
void get_hard_examples(const string & output_dir, const string & svm_path, const string & dir,
	const string & names, Size& size);

const Size cell_size(96, 160);