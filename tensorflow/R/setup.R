# %% TensorFlow values used by the equations
# Core runtime only. Public datasets/models are loaded by their experiment.
suppressPackageStartupMessages(library(reticulate))
tf <- import("tensorflow", convert = FALSE)
for(device in py_to_r(tf$config$list_physical_devices("GPU")))
  try(tf$config$experimental$set_memory_growth(device,TRUE),silent=TRUE)
# The correctness-first port uses float32 arithmetic, not implicit TF32. The
# historical accelerated path is an execution ablation after reproduction.
tf$config$experimental$enable_tensor_float_32_execution(FALSE)
np <- import("numpy", convert = FALSE)
as_r <- function(x) if (inherits(x, "python.builtin.object")) py_to_r(x) else x
value <- function(x) as_r(x$numpy())
shape <- function(x) vapply(as_r(x$shape$as_list()),function(d)
  if(is.null(d))NA_integer_ else as.integer(d),integer(1))
scalar <- function(x, like) if(inherits(x,"python.builtin.object"))tf$cast(x,like$dtype) else tf$constant(x,dtype=like$dtype)
zero <- function(x) tf$zeros_like(x)
# Embedding gathers return IndexedSlices: repeated token indices must SUM before
# squaring gradients or applying the dense Gaussian optimizer recurrence.
dense_gradients <- function(xs) lapply(as_r(xs),tf$convert_to_tensor)
# Selected instructional figures; all numerical results remain in the package.
TUTORIAL_FIGURES <- c(
 "tutorial_gaussian_mlp_exact_message_equivalence",
 "tutorial_shared_ep_message_equivalence","tutorial_shared_mfvi_message_equivalence",
 "tutorial_forward_accuracy_cost","tutorial_forward_mc_agreement",
 "tutorial_backward_accuracy_cost","tutorial_backward_mc_agreement",
 "matched_forward","matched_backward","matched_rank",
 "mnist_architectures","mnist_mlp_clean_uncertainty","mnist_lenet_clean_uncertainty",
 "mnist_mlp_clean_learning","mnist_lenet_clean_learning",
 "mnist_lenet_dirty_uncertainty","mnist_dirty_ambiguity",
 "tutorial_resnet20_cifar10_calibration_and_selection",
 "tutorial_densenet40_cifar10_calibration_and_selection",
 "tutorial_sst2_calibration_and_selection","tutorial_clinc_calibration_and_selection",
 "resnet20_cifar10_new_R_training_seed41_learning")

