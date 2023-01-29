주피터 노트북 파이썬 스크립트 전환: 
`jupyter nbconvert --to python \
  --output-dir <output_dir> --output <filename> \
  <notebook_path>`

- 케라스 모델 훈련 중
"Node: 'StatefulPartitionedCall_11'
could not find registered platform with id: 0x176f98af0
	 [[{{node StatefulPartitionedCall_11}}]] [Op:__inference_train_function_5258]"
란 에러 메시지가 뜬다면:
(출처: https://developer.apple.com/forums/thread/721735)
Is this on the latest wheels with tensorflow-macos==2.11 and tensorflow-metal==0.7.0? In that case this most probably has to do with recent changes on tensorflow side for version 2.11 where a new optimizer API has been implemented where a default JIT compilation flag is set (https://blog.tensorflow.org/2022/11/whats-new-in-tensorflow-211.html). This is forcing the optimizer op to take an XLA path that the pluggable architecture has not implemented yet causing the inelegant crash as it cannot fall back to supported operations. Currently the workaround is to use the older API for optimizers that was used up to TF 2.10 by exporting it from the .legacy folder of optimizers. So more concretely by using Adam optimizer as an example one should change:
`from tensorflow.keras.optimizers import Adam` to `from tensorflow.keras.optimizers.legacy import Adam`.
This should restore previous behavior while the XLA path support is being worked on. Let me know if this solves the issue for you! And if not could you let us know which tf-macos and tf-metal versions you are seeing this and a script I can use to reproduce the issue?