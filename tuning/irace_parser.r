library("tibble")
library("dplyr")
setwd("./tuning/")

args <- commandArgs()
algorithm <- ifelse(!is.na(args[6]), args[6], "lns")
instance_size <- ifelse(!is.na(args[7]), args[7], "50")
irace_file <- file.path("..", "data", "tuning", instance_size, "train", algorithm, "irace")
irace_res <- get(load(paste0(irace_file, ".Rdata")))

# obtain parameter labels of algorithm
irace_params <- irace_res$scenario$parameters$.params
irace_params_len <- length(irace_params)
irace_labels <- vector(, irace_params_len)
for (par_idx in 1:irace_params_len) {
  label_parts <- strsplit(trimws(irace_params[[par_idx]]$label), "-")[[1]]
  label_parts <- label_parts[label_parts != ""]
  label <- paste(label_parts, collapse = "_")
  irace_labels[par_idx] <- label
}

# obtain average cost and runtimes of configs
irace_exps <- irace_res$state$experiment_log
irace_exps <- irace_exps %>%
  group_by(configuration) %>%
  summarise(median_score = median(cost), median_time = median(time)) %>%
  select(-configuration)

# obtain complete tuning data
irace_df <- irace_res$allConfigurations
irace_df <- irace_df %>%
  select(-one_of(".ID.", ".PARENT."))  %>%
  setNames(irace_labels) %>%
  add_column(instance_size = as.numeric(instance_size), .before = 1) %>%
  bind_cols(irace_exps) %>%
  arrange(median_score)

# store parsed tuning csv
write.csv(irace_df, paste0(irace_file, ".csv"), row.names = FALSE, quote=FALSE)