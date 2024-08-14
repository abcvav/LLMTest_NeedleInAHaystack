python needlehaystack.prepare_data \
  --context_lengths_min 2000 \
  --context_lengths_max 32000 \
  --context_lengths_step 1000 \
  --document_depth_percent_min 0 \
  --document_depth_percent_max 100 \
  --document_depth_percent_step 5 \
  --final_context_length_buffer 200 \
  --multi_needle False \
  --tokenizer_vocab_file ""