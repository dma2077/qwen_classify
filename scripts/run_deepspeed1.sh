sleep 1800
nohup bash /llm_reco/dehua/code/qwen_classify/scripts/run_deepspeed_ls_3e_6_ls.sh > /llm_reco/dehua/code/qwen_classify/output_deepspeed_3e_6_ls.log 2>&1
nohup bash /llm_reco/dehua/code/qwen_classify/scripts/run_deepspeed_ls_3e_6.sh > /llm_reco/dehua/code/qwen_classify/output_deepspeed_3e_6.log 2>&1
