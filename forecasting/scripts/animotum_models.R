scriptPath <- getwd()
RootDir <- normalizePath(dirname(scriptPath), winslash = "/", mustWork = TRUE) # Root directory
source(file.path(RootDir, "animotum.R"))

# defaults
model <- "rw"
task <- "forecasting"
individual_IDS <- FALSE
test_partition <- "test"
pf_only <- FALSE
test_days <- 7
baseline <- FALSE

# parse args
args <- commandArgs(trailingOnly = TRUE)

for (i in seq(1, length(args), 2)) {
    option <- args[i]
    value <- args[i + 1]

    if (option == "--model" || option == "-m") {
        model <- value
    } else if (option == "--task" || option == "-t") {
        task <- value
    } else if (option == "--individual-ids" || option == "-i") {
        if (value == "TRUE" || value == "true") {
            individual_IDS <- TRUE
        } else if (value == "FALSE" || value == "false") {
            individual_IDS <- FALSE
        } else {
            stop("Invalid value for --individual-ids")
        }
    } else if (option == "--test-partition" || option == "-p") {
        test_partition <- value
    } else if (option == "--pf-only" || option == "-P") {
        if (value == "TRUE" || value == "true") {
            pf_only <- TRUE
        } else if (value == "FALSE" || value == "false") {
            pf_only <- FALSE
        } else {
            stop("Invalid value for --pf-only")
        }
    } else if (option == "--test-days" || option == "-d") {
        test_days <- as.integer(value)
    } else if (option == "--baseline" || option == "-b") {
        if (value == "TRUE" || value == "true") {
            baseline <- TRUE
        } else if (value == "FALSE" || value == "false") {
            baseline <- FALSE
        } else {
            stop("Invalid value for --baseline")
        }
    } else {
        stop("Invalid option: ", option)
    }
}

if (pf_only) {
    result = train_and_store(model=model, task=task, test_partition=test_partition, pf_only=TRUE, test_days=test_days, baseline=baseline)
} else{
    if (individual_IDS == "TRUE") {
        individual_IDS <- TRUE
    }

    if (individual_IDS) {
        printf("Using individual IDs\n")
        func = train_and_store_single_ID_all
    } else {
        printf("Using group IDs\n")
        func = train_and_store
    }

    result = func(model=model, task=task, test_partition=test_partition)
}

printf("DONE!")
