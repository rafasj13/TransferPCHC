install.packages("https://www.bnlearn.com/releases/bnlearn_latest.tar.gz", repos = NULL, type = "source")
library(bnlearn)


modify_bn_structure <- function(
    rds_path,
    output_dir,
    percent_to_modify = 0.2,
    sample_size = 50000,
    seed = 123
) {
  # Create output directory if it doesn't exist
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  
  # Load network
  net_fit <- readRDS(rds_path)
  net_dag <- bn.net(net_fit)
  net_name <- tools::file_path_sans_ext(basename(rds_path))
  
  # Sample from original network
  set.seed(seed)
  sampled <- rbn(net_fit, n = sample_size)
  arcs_orig <- as.data.frame(arcs(net_dag))
  write.csv(sampled, file.path(output_dir, paste0(net_name, "_sampled.csv")), row.names = FALSE)
  write.csv(arcs_orig, file.path(output_dir, paste0(net_name, "_arcs.csv")), row.names = FALSE)
  
  # Remove percentage of arcs
  n_arcs <- nrow(arcs_orig)
  n_modify <- ceiling(n_arcs * percent_to_modify)
  arcs_to_remove <- arcs_orig[sample(n_arcs, n_modify), ]
  net_mod <- net_dag
  for(i in 1:nrow(arcs_to_remove)) {
    net_mod <- drop.arc(net_mod, from = arcs_to_remove[i, "from"], to = arcs_to_remove[i, "to"])
  }
  
  # Add new arcs without cycles
  nodes <- nodes(net_mod)
  possible_arcs <- expand.grid(from = nodes, to = nodes, stringsAsFactors = FALSE)
  possible_arcs <- possible_arcs[possible_arcs$from != possible_arcs$to, ]
  existing_arcs <- paste(arcs_orig$from, arcs_orig$to)
  possible_arcs <- possible_arcs[!paste(possible_arcs$from, possible_arcs$to) %in% existing_arcs, ]
  
  n_add <- n_modify
  added <- 0
  tries <- 0
  while (added < n_add && tries < 10 * n_add && nrow(possible_arcs) > 0) {
    idx <- sample(nrow(possible_arcs), 1)
    from <- possible_arcs$from[idx]
    to <- possible_arcs$to[idx]
    try_result <- try({
      net_test <- set.arc(net_mod, from = from, to = to)
      net_mod <- net_test
      added <- added + 1
    }, silent = TRUE)
    possible_arcs <- possible_arcs[-idx, ]
    tries <- tries + 1
  }
  
  # Compare structures
  arcs_mod <- as.data.frame(arcs(net_mod))
  added_arcs <- setdiff(
    paste(arcs_mod$from, arcs_mod$to),
    paste(arcs_orig$from, arcs_orig$to)
  )
  removed_arcs <- setdiff(
    paste(arcs_orig$from, arcs_orig$to),
    paste(arcs_mod$from, arcs_mod$to)
  )
  added_arcs_df <- arcs_mod[paste(arcs_mod$from, arcs_mod$to) %in% added_arcs, ]
  removed_arcs_df <- arcs_orig[paste(arcs_orig$from, arcs_orig$to) %in% removed_arcs, ]
  
  cat("Number of arcs in original:", nrow(arcs_orig), "\n")
  cat("Number of arcs in modified:", nrow(arcs_mod), "\n")
  cat("Number of arcs added:", nrow(added_arcs_df), "\n")
  cat("Number of arcs removed:", nrow(removed_arcs_df), "\n")
  cat("\nAdded arcs:\n")
  print(added_arcs_df)
  cat("\nRemoved arcs:\n")
  print(removed_arcs_df)
  
  distance <- hamming(net_mod, net_dag)
  cat("Structural Hamming Distance (SHD):", distance, "\n")
  
  # Fit and sample from modified network
  net_mod_fit <- bn.fit(net_mod, data = sampled)
  new_samples <- rbn(net_mod_fit, n = sample_size)
  write.csv(new_samples, file.path(output_dir, paste0(net_name, "_", percent_to_modify*100, "p_sampled.csv")), row.names = FALSE)
  write.csv(arcs_mod, file.path(output_dir, paste0(net_name, "_", percent_to_modify*100, "p_arcs.csv")), row.names = FALSE)
  
  invisible(list(
    original_arcs = arcs_orig,
    modified_arcs = arcs_mod,
    added_arcs = added_arcs_df,
    removed_arcs = removed_arcs_df,
    SHD = distance,
    original_samples = sampled,
    modified_samples = new_samples
  ))
}

percentages <- c(0.05, 0.1, 0.2)
for (p in percentages) {
modify_bn_structure("bnlearn_nets/arth150.rds", "bnlearn_nets/arth150/", percent_to_modify = p, sample_size = 50000, seed=15)
}
