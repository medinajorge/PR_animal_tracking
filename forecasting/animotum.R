library(chron)
library(dplyr)
library(lubridate)
library(purrr)
library(tools)
library(progress)
library(R.utils)
library(sf)
library(tidyr)
library(aniMotum)
library(trip)

# Sources:
# https://github.com/ianjonsen/aniMotum/blob/master/R/fit_ssm.R
# https://ianjonsen.github.io/aniMotum/articles/SSM_fitting.html
# NOTE: simulated trajectories from the ssm model are not valid for uncertainty estimation.
# https://ianjonsen.github.io/aniMotum/articles/Track_simulation.html

DataDir = "data"

load_data = function(test_partition = "test"){
  cat("Loading data...\n")
  if (test_partition == "test"){
    train_dataset = rbind(read.csv(paste0(DataDir, "/train.csv")),
                          read.csv(paste0(DataDir, "/val.csv")))
    test_dataset = read.csv(paste0(DataDir, "/test.csv"))
  } else if (test_partition == "val"){ # for validation
    cat("Using validation data as test data...\n")
    train_dataset = read.csv(paste0(DataDir, "/train.csv"))
    test_dataset = read.csv(paste0(DataDir, "/val.csv"))
  } else {
    stop("test_partition must be either 'test' or 'val'")
  }
  train_dataset = format_data(train_dataset, id='ID')
  test_dataset = format_data(test_dataset, id='ID')

  cat("Removing duplicates...\n")
  init_dim_train = dim(train_dataset)[1]
  init_dim_test = dim(test_dataset)[1]
  train_dataset = train_dataset[!duplicated(train_dataset),]
  test_dataset = test_dataset[!duplicated(test_dataset),]
  cat(paste0("Removed ", init_dim_train - dim(train_dataset)[1], " duplicates from train dataset\n"))
  cat(paste0("Removed ", init_dim_test - dim(test_dataset)[1], " duplicates from test dataset\n"))

  return(list(train_dataset=train_dataset, test_dataset=test_dataset))
}

days_to_seconds = function(days){
    return(days * 24 * 3600)
}

hours_to_seconds = function(hours){
    return(hours * 3600)
}

bin_data = function(df){
  initial_date = df$date[[1]]
  end_date = df$date[[dim(df)[1]]]
  initial_year = format(initial_date, "%Y")
  end_year = format(end_date, "%Y")
  jan_1 = as.POSIXct(paste0(initial_year, "-01-01"), tz = "UTC")
  jan_1_two_years_later = as.POSIXct(paste0(as.numeric(end_year) + 1, "-01-01"), tz = "UTC")
  time_bins = seq(jan_1, jan_1_two_years_later, by = "6 hours")
  # average observations every 6 hours
  df_binned = df %>%
      mutate(time_bin = cut(date, breaks = time_bins, labels = FALSE))

  df_binned = df_binned %>%
    group_by(time_bin) %>%
    summarise(lat = mean(lat),
              # Conditional lon calculation
              lon = if(any(lon < 0) & any(lon > 0)) median(lon) else mean(lon),
              date = time_bins[[time_bin[1]]],
              lc = lc[1],
              id = id[1])
  df_binned = format_data(df_binned)
  return(df_binned)
}

preprocess_forecasting = function(df, test_days = 7, baseline = FALSE){
  if (baseline){
    test_days = test_days*4
  }
  final_date = df$date[[dim(df)[1]]]
  train_date = final_date - ( days_to_seconds(test_days) - hours_to_seconds(6) )
  train = df[df$date <= train_date,]
  test = df[df$date > train_date & df$date <= final_date,]
  test = bin_data(test)
  return(list(train=train, test=test))
}

preprocess_imputation = function(df, training_days = 28, test_days = 7, baseline = FALSE){
  if (baseline){
    return(list(train=df,
              test=bin_data(df)))
  } else {
    train_dfs = list()
    test_dfs = list()
    initial_date = df$date[[1]]
    final_date = df$date[[dim(df)[1]]]
    train_date = initial_date + days_to_seconds(training_days)
    test_init = train_date
    test_date = test_init + days_to_seconds(test_days) - hours_to_seconds(6)
    final_test_init = final_date - days_to_seconds(test_days + training_days/2) # leave at least half of the training days after the imputation window.
    i = 1
    while (test_init < final_test_init) {
        train = df[df$date > initial_date & df$date <= train_date,]
        test = df[df$date > test_init & df$date <= test_date,]
        if (dim(test)[1] > test_days){# at least one observation / day on average
          test = bin_data(test)
          train_dfs[[i]] = train
          test_dfs[[i]] = test
        }
        initial_date = test_date
        train_date = initial_date + days_to_seconds(training_days)
        test_init = train_date
        test_date = test_init + days_to_seconds(test_days) - hours_to_seconds(6)
        i = i + 1
    }
    # append remaining data to train
    remaining_data = df[df$date > initial_date & df$date <= final_date,]
    train_dfs[[i]] = remaining_data
    if (baseline){
      test_dfs[[i]] = bin_data(remaining_data)
    }
    return(list(train=do.call(rbind, train_dfs),
                test=do.call(rbind, test_dfs)))
  }
}

# preprocess_imputation = function(df, training_days = 28, test_days = 7, baseline = FALSE){
#   train_dfs = list()
#   test_dfs = list()
#   initial_date = df$date[[1]]
#   final_date = df$date[[dim(df)[1]]]
#   train_date = initial_date + days_to_seconds(training_days)
#   baseline_padding = days_to_seconds(test_days-2)
#   if (baseline){
#     test_init = train_date - baseline_padding
#   }
#   else{
#     test_init = train_date
#   }
#   test_date = test_init + days_to_seconds(test_days) - hours_to_seconds(6)
#   if (baseline){
#     test_date = test_date + 2*baseline_padding
#   }
#   final_test_init = final_date - days_to_seconds(test_days + training_days/2) # leave at least half of the training days after the imputation window.
#   i = 1
#   while (test_init < final_test_init) {
#       train = df[df$date > initial_date & df$date <= train_date,]
#       test = df[df$date > test_init & df$date <= test_date,]
#       if (dim(test)[1] > test_days){# at least one observation / day on average
#         test = bin_data(test)
#         train_dfs[[i]] = train
#         test_dfs[[i]] = test
#       }
#       if (baseline){
#         initial_date = test_date - baseline_padding
#       } else {
#         initial_date = test_date
#       }
#       train_date = initial_date + days_to_seconds(training_days)
#       if (baseline){
#         test_init = train_date - baseline_padding
#       }
#       else{
#         test_init = train_date
#       }
#       test_date = test_init + days_to_seconds(test_days) - hours_to_seconds(6)
#       if (baseline){
#         test_date = test_date + 2*baseline_padding
#       }
#       i = i + 1
#   }
#   # append remaining data to train
#   remaining_data = df[df$date > initial_date & df$date <= final_date,]
#   train_dfs[[i]] = remaining_data
#   if (baseline){
#     test_dfs[[i]] = bin_data(remaining_data)
#   }
#   return(list(train=do.call(rbind, train_dfs),
#               test=do.call(rbind, test_dfs)))
# }

test_train_split = function(test_dataset, test_days = 7, training_days = 28, task = "forecasting", baseline = FALSE){
  if (task == "forecasting"){
    pp_func = function(x) preprocess_forecasting(x, test_days = test_days, baseline = baseline)
  }
  else if (task == "imputation"){
    pp_func = function (x) preprocess_imputation(x, test_days = test_days, training_days = training_days, baseline = baseline)
  }
  else{
    stop("task must be either 'forecasting' or 'imputation'")
  }
  cat(paste0("Splitting test dataset in train and test with mode: ", task, '\n'))
  train = list()
  test = list()
  unique_IDs = unique(test_dataset$id)
  for (ID in unique_IDs){
    df_ID = test_dataset[test_dataset$id == ID,]
    N = dim(df_ID)[1]
    if (N > 2){
      initial_date = df_ID$date[[1]]
      min_final_date = initial_date + days_to_seconds(training_days + test_days)
      final_date = df_ID$date[[N]]
      if (final_date >= min_final_date){ # assert there is enough data.
        pp = pp_func(df_ID)
        train[[ID]] = pp$train
        test[[ID]] = pp$test
      }
    }
  }
  return(list(train=do.call(rbind, train),
              test=do.call(rbind, test)))
}

preprocess_ssm = function(training_days = 28, test_days = 7, task = "forecasting", test_partition = "test", baseline = FALSE){
  data = load_data(test_partition = test_partition)
  test_pp = test_train_split(data$test_dataset, training_days = training_days, test_days = test_days, task = task, baseline = baseline)
  train = rbind(data$train_dataset, test_pp$train)
  test = test_pp$test
  return(list(train=train, test=test))
}

train_and_store = function(model = "rw", vmax = NA, ang = NA, distlim=NA,
                           prune = TRUE, pf_only = FALSE, partition = "test", test_partition = "test", baseline = FALSE,
                           training_days = 28, test_days = 7, task = "forecasting"){
    data = preprocess_ssm(training_days=training_days, test_days=test_days, task=task, test_partition=test_partition, baseline=baseline)
    if (prune) {
        cat("Pruning data...\n")
        IDs = get_valid_IDs(model, task, test_partition)
        data$train = data$train[data$train$id %in% IDs,]
        data$test = data$test[data$test$id %in% IDs,]
        prune_str = 'pruned'
    } else {
        prune_str = ''
    }
    if (is.na(vmax)){
        vmax = 10 #km / h. Also migrate 33800km in 365 days = 92.6km/day. Source: https://oceanwide-expeditions.com/to-do/wildlife/elephant-seal
        vmax = vmax * 1000 / 3600 # m/s
        cat(paste0("Using default value for velocity of Southern elephant seal: ", vmax, '\n'))
    }
    if (pf_only){
        cat("Pre-filtering test data only...\n")
        ssm_test = prefilter_data(data, vmax=vmax, ang=ang, distlim=distlim, partition=partition)
        if (baseline){
            SaveDir = paste0(DataDir, "/", 'baseline')
        } else {
            SaveDir = paste0(DataDir, "/", 'pf_only')
        }
    } else {
        time.step = get_time_step_prediction(data)
        cat("Training model...\n")
        fit = fit_ssm(data$train, model = model, vmax=vmax, ang=ang, distlim=distlim, time.step=time.step)
        if (partition == "train"){
          to_grab = "fitted"
        } else if (partition == "test"){
          to_grab = "predicted"
        } else {
          stop("partition must be either 'train' or 'test'")
        }
        ssm <- grab(fit, to_grab, as_sf = FALSE, normalise = TRUE, group = FALSE)
        test_IDs = unique(data$test$id)
        ssm_test = ssm[ssm$id %in% test_IDs,]
        cat("Storing data...\n")
        SaveDir = paste0(DataDir, "/", model)
    }
    # create directory
    if (!dir.exists(SaveDir)) {
      dir.create(SaveDir, recursive = TRUE)
    }
    if (test_days == 7){
      filepath = paste0(SaveDir, "/", partition, "_", "test-partition-", test_partition, "_", task, "_", prune_str, ".csv")
    } else {
      filepath = paste0(SaveDir, "/", partition, "_", "test-partition-", test_partition, "_", task, "_", prune_str, "_", test_days, "-days.csv")
    }
    write.csv(ssm_test, file = filepath, row.names = FALSE)
    cat(paste0("Data stored in: ", filepath, '\n'))
    cat("NOTE!! To obtain the Confidence Regions multiply the SD: 95% (x1.96), 90% (x1.645), 50% (x0.674)\n") # TODO: Add this to the documentation. Check the factors.
    if (!pf_only){
        return(ssm)
    } else {
        return(ssm_test)
    }
}

train_and_store_single_ID_all = function(model = "rw", vmax = NA, ang = NA, distlim=NA, # function to be used to see which fits can be done
                           training_days = 28, test_days = 7, task = "forecasting", test_partition = "test"){
  data = preprocess_ssm(training_days=training_days, test_days=test_days, task=task, test_partition=test_partition)
  train = data$train
  test = data$test
  if (is.na(vmax)){
      vmax = 10 #km / h. Also migrate 33800km in 365 days = 92.6km/day. Source: https://oceanwide-expeditions.com/to-do/wildlife/elephant-seal
      vmax = vmax * 1000 / 3600 # m/s
      cat(paste0("Using default value for velocity of Southern elephant seal: ", vmax, '\n'))
  }

  IDs = unique(train$id)
  pb = txtProgressBar(min = 0, max = length(IDs), style = 3)
  for (ID in IDs){
    data_ID = train[train$id == ID,]
    # if Id in data$test, set time step from data$test, else set to 6
    if (ID %in% test$id){
      time.step = test[test$id == ID, c("id", "date")]
    } else {
      time.step = 6
    }
    # try to fit the model
    tryCatch({
      fit = fit_ssm(data_ID, model = model, vmax=vmax, ang=ang, distlim=distlim, time.step=time.step)
      ssm <- grab(fit, "predicted", as_sf = FALSE, normalise = TRUE, group = FALSE)
    }, error = function(e){
      cat(paste0("Error fitting model for ID: ", ID, ". Error: ", e$message, '\nReturning empty dataframe\n'))
      # ssm empty dataframe
      ssm = data.frame()
    })
    SaveDir = paste0(DataDir, "/", model, "/single_ID")
    # create directory
    if (!dir.exists(SaveDir)) {
      dir.create(SaveDir, recursive = TRUE)
    }
    filepath = paste0(SaveDir, "/", test_partition, "_", task, "_", ID, ".csv")
    write.csv(ssm, file = filepath, row.names = FALSE)
    cat(paste0("Data stored in: ", filepath, '\n'))
    setTxtProgressBar(pb, ID)
  }
  close(pb)
}


get_valid_IDs = function(model, task, test_partition = "test"){
  modelDir = paste0(DataDir, "/", model, '/single_ID')
  files = list.files(modelDir, full.names = TRUE)
  IDs = c()

  for (file in files){
    filename = basename(file)
    if (grepl(task, filename) & grepl(test_partition, filename)){
      df = read.csv(file)
      if (dim(df)[1] > 1) {
        ID_and_ext = strsplit(filename, "_")[[1]][3]
        ID = strsplit(ID_and_ext, "\\.")[[1]][1]
        IDs = c(IDs, ID)
      }
    }
  }
  return(IDs)
}

get_time_step_prediction = function(data){
  # returns the test dates and IDs and null values for the rest of the IDs
  IDs_not_in_test = setdiff(unique(data$train$id), unique(data$test$id))
  dfs = list()
  for (ID in IDs_not_in_test) {
    data_ID = data$train[data$train$id == ID,]
    last_date = max(data_ID$date)
# add 4 hours
    date_predict = last_date + c(1, 2) * 4 * 3600
# create dataframe with columns id and date
    df = data.frame(id=ID, date=date_predict)
    dfs[[ID]] = df
  }
  df_full = do.call(rbind, dfs)
  time.step = rbind(data$test[, c("id", "date")], df_full)
  return(time.step)
}


pf_dup_dates <- function(x, min.dt) {

  x$keep <- difftime(x$date, c(as.POSIXct(NA), x$date[-nrow(x)]),
                     units = "secs") > min.dt

  x$keep <- ifelse(is.na(x$keep), TRUE, x$keep)

  return(x)
}

pf_obs_type <- function(x) {

  ## determine observation type: LS, KF, GPS or GLS
  x$obs.type <- NA
  x$obs.type <- with(x,
                     ifelse(!is.na(smaj) & !is.na(smin) & !is.na(eor),
                            "KF", obs.type))
  x$obs.type <- with(x, ifelse(lc %in% c(3,2,1,0,"A","B","Z") &
                                 (is.na(smaj) | is.na(smin) | is.na(eor)),
                               "LS", obs.type))
  x$obs.type <- with(x, ifelse(lc == "G" &
                                 (is.na(smaj) | is.na(smin) |is.na(eor)),
                               "GPS", obs.type))
  x$obs.type <- with(x, ifelse(lc == "GL" &
                                 (is.na(smaj) | is.na(smin) | is.na(eor)) &
                                 (!is.na(x.sd) & !is.na(y.sd)),
                               "GL", obs.type))

  ##  if any records with smaj/smin = 0 then set to NA and obs.type to "LS"
  ##  convert error ellipse smaj & smin from m to km and eor from deg to rad
  x$smaj <- with(x, ifelse(smaj == 0 | smin == 0, NA, smaj)) / 1000
  x$smin <- with(x, ifelse(smin == 0 | is.na(smaj), NA, smin)) / 1000
  x$eor <- with(x, ifelse(is.na(smaj) & is.na(smin), NA, eor)) / 180 * pi

  x$obs.type <- with(x, ifelse(is.na(smaj) & is.na(smin) & is.na(eor) &
                                 (obs.type != "GL" & obs.type != "GPS"),
                               "LS", obs.type))

  if(all("lon" %in% names(x), "lat" %in% names(x), "lonerr" %in% names(x), "laterr" %in% names(x))) {
    ## if GL SD's are loneer/laterr then convert from deg to km
    x$x.sd <- with(x, x.sd * 6378.137 / 180 * pi)
    x$y.sd <- with(x, y.sd * 6378.137 / 180 * pi)

  } else if(all((all("lon" %in% names(x), "lat" %in% names(x)) |
             all("x" %in% names(x), "y" %in% names(x))), "x.sd" %in% names(x), "y.sd" %in% names(x))){
    ## if GL SD's are x.sd/y.sd then convert from m to km
    x$x.sd <- with(x, x.sd / 1000)
    x$y.sd <- with(x, y.sd / 1000)
  }

  return(x)
}

pf_sda_filter <- function(x, spdf, vmax, ang, distlim) {
## Use internal version of trip::sda to identify outlier locations
if (spdf) {
  if(inherits(x, "sf") && st_is_longlat(x)) {

    xy <- as.data.frame(st_coordinates(x))
    names(xy) <- c("lon","lat")
    x <- cbind(x, xy)

  } else if(inherits(x, "sf") && !st_is_longlat(x)) {

    xy <- st_transform(x, crs = st_crs("+proj=longlat +datum=WGS84 +no_defs"))
    xy <- as.data.frame(st_coordinates(xy))
    names(xy) <- c("lon","lat")
    x <- cbind(x, xy)
  }

  ## was req'd when using trip::sda - keep in case we want to revert now that
  ##  {trip} has been updated and 'un-archived'
#  x.tr <- subset(x, keep)[, c("lon","lat","date","id","lc","smaj","smin",
#                              "eor","lonerr","laterr","keep","obs.type")]
#  names(x.tr)[1:2] <- c("x","y")
#  x.tr <- suppressWarnings(trip(as.data.frame(x.tr), TORnames = c("date", "id"),
#                                correct_all = FALSE))
  x.tr <- subset(x, keep)
  p.GL <- sum(x$obs.type == "GL") / nrow(x)
  if(p.GL > 0.75) {
    filt <- "spd"
  } else {
    filt <- "sda"
  }

  if(any(is.na(ang))) ang <- c(0,0)
  if(any(is.na(distlim))) distlim <- c(0,0)
  trip.dat <- suppressWarnings(with(x.tr, trip(data.frame(lon, lat, tms = date, id),
                                                     correct_all = FALSE)))

  if (filt == "sda") {
    tmp <-
      suppressWarnings(try(sda(trip.dat,
                               smax = vmax * 3.6,
                               # convert m/s to km/h
                               ang = ang,
                               distlim = distlim / 1000),     # convert m to km
                               silent = TRUE))
                       ## screen potential sdafilter errors
                       if (inherits(tmp, "try-error")) {
                         warning(
                           paste(
                             "\ntrip::sda produced an error on id",
                             x$id[1],
                             "using trip::speedfilter instead"
                           ),
                           immediate. = TRUE
                         )

                         tmp <-
                           suppressWarnings(try(speedfilter(trip.dat,
                                                            max.speed = vmax * 3.6),    # convert m/s to km/h
                                                            silent = TRUE))

                                            if (inherits(tmp, "try-error")) {
                                              warning(
                                                paste(
                                                  "\ntrip::speedfilter also produced an error on id",
                                                  x$id[1],
                                                  "can not apply speed filter prior to SSM filtering"
                                                ),
                                                immediate. = TRUE
                                              )
                                            }
                       }
  } else if (filt == "spd") {
    tmp <-
      suppressWarnings(try(speedfilter(x.tr, max.speed = vmax * 3.6),
                           silent = TRUE))

    if (inherits(tmp, "try-error")) {
      warning(
        paste(
          "\ntrip::speedfilter also produced an error on id",
          x$id[1],
          "can not apply speed filter prior to SSM filtering"
        ),
        immediate. = TRUE
      )
    }
  }

  x[x$keep, "keep"] <- tmp

}

  return(x)
}

wrap_lon <- function(lon, lon_min = -180) {

  (lon - lon_min) %% 360 + lon_min

}

pf_sf_project <- function(x) {

  if(!inherits(x, "sf")) {
    ##  if lon spans -180,180 then shift to
    ##    0,360; else if lon spans 360,0 then shift to
    ##    -180,180 ... have to do this on keep subset only

    xx <- subset(x, keep)

    if("lon" %in% names(x)) {
      coords <- c("lon", "lat")
      sf_locs <- st_as_sf(x, coords = coords,
                          crs = st_crs("+proj=longlat +datum=WGS84 +no_defs"))

      if (any(diff(wrap_lon(xx$lon, 0)) > 300)) {
        prj <- "+proj=merc +lon_0=0 +datum=WGS84 +units=km +no_defs"
      } else if (any(diff(wrap_lon(xx$lon,-180)) < -300) ||
                 any(diff(wrap_lon(xx$lon,-180)) > 300)) {
        prj <- "+proj=merc +lon_0=180 +datum=WGS84 +units=km +no_defs"
      } else {
        prj <- "+proj=merc +lon_0=0 +datum=WGS84 +units=km +no_defs"
      }

      sf_locs <-  st_transform(sf_locs, st_crs(prj))

    } else {
      coords <- c("x", "y")
      sf_locs <- st_as_sf(x, coords = coords,
                          crs = st_crs("+proj=merc +units=m +datum=WGS84 +no_defs"))
      prj <- st_crs(sf_locs)$input
      sf_locs <- st_transform(sf_locs, sub("units=m", "units=km", prj, fixed = TRUE))
    }

  } else {
    ## if input data projection is longlat then set prj merc, otherwise respect
    ##     user-supplied projection
    if(st_is_longlat(x)) {
      prj <- "+proj=merc +lon_0=0 +datum=WGS84 +units=km +no_defs"
    } else {
      prj <- st_crs(x)$input
    }

    # if data CRS units are m then change to km, otherwise optimiser may choke
    if (grepl("units=m", prj, fixed = TRUE)) {
      message("Converting projection units from m to km for efficient optimization")
      prj <- sub("units=m", "units=km", prj, fixed = TRUE)
    }
    ll <- which(names(x) %in% c("lon","lat"))
    sf_locs <- x[, -ll]
    sf_locs <- st_transform(sf_locs, prj)
  }

  return(sf_locs)
}

pf_add_emf <- function(x, emf) {

  ## add LS error info to corresponding records
  ## set emf's = NA if obs.type %in% c("KF","GL") - not essential but for clarity
  if(is.null(emf)) {
    tmp <- emf()
  } else if(is.data.frame(emf)) {
    tmp <- emf
  }

  x$lc <- with(x, as.character(lc))
  x <- merge(x, tmp, by = "lc", all.x = TRUE, sort = FALSE)

  if(all("lonerr" %in% names(x), "laterr" %in% names(x))) {
    x <- x[order(x$date), c("id","date","lc","smaj","smin","eor",
                            "lonerr","laterr","keep","obs.type",
                            "emf.x","emf.y","geometry")]
  } else {
    x <- x[order(x$date), c("id","date","lc","smaj","smin","eor",
                            "x.sd","y.sd","keep","obs.type",
                            "emf.x","emf.y","geometry")]
  }

  x$emf.x <- with(x, ifelse(obs.type %in% c("KF","GL"), NA, emf.x))
  x$emf.y <- with(x, ifelse(obs.type %in% c("KF","GL"), NA, emf.y))

  if (sum(is.na(x$lc)) > 0)
    stop(
      "\n NA's found in location class values"
    )

  return(x)
}

prefilter <-
  function(x,
           vmax = 5,
           ang = c(15,25),
           distlim = c(2500, 5000),
           spdf = TRUE,
           min.dt = 0,
           emf = NULL
           ) {

    ## check args
    if(!(is.numeric(vmax) & vmax > 0))
      stop("vmax must be a positive, non-zero value representing an upper speed threshold in m/s")
    if(!any((is.numeric(ang) & length(ang) == 2) || is.na(ang)))
      stop("ang must be either a vector of c(min, max) angles in degrees defining extreme steps to be removed from trajectory, or NA")
    if(!any((is.numeric(distlim) & length(distlim) == 2) || is.na(distlim)))
      stop("distlim must be either a vector of c(min, max) in m defining distances of extreme steps to be removed from trajectory, or NA")
    if(!is.logical(spdf)) stop("spdf must either TRUE to turn on, or FALSE to turn off speed filtering")
    if(!is.na(min.dt) & !(is.numeric(min.dt) & min.dt >= 0)) stop("min.dt must be a positive, numeric value representing the minimum time difference between observed locations in s")
    if(!any(is.null(emf) || (is.data.frame(emf) & nrow(emf) > 0)))
      stop("emf must be either NULL to use default emf (type emf() to see values), or a data.frame (see ?emf for details")

    if(length(unique(x$id)) > 1)
      stop("Multiple individual tracks in Data, use `fit_ssm(..., pf = TRUE)`")

    ##  1. flag duplicate date (delta(date) < min.dt) records
    x <- pf_dup_dates(x, min.dt)

    ##  2. determine observation type: LS, KF, GPS or GLS
    x <- pf_obs_type(x)

    ##  3. identify extreme locations with a speed/distance/angle filter
    x <- pf_sda_filter(x, spdf, vmax, ang, distlim)

    ##  4. project from longlat to merc or respect user-supplied projection &
    ##       ensure that longitudes straddling -180,180 or 0,360 are shifted
    ##       appropriately
    x <- pf_sf_project(x)

    ##  5. add location error multiplication factors and finalise data structure
    ##      for use by sfilter()
    x <- pf_add_emf(x, emf)

    return(x)
}

prefilter_data = function(data, vmax, ang, distlim, spdf=TRUE, min.dt=0, emf=NULL, partition="test"){
  if (partition == "test"){
    x = data$test
  } else if (partition == "train"){
    x = data$train
  } else {
    stop("partition must be either 'test' or 'train'")
  }
  ## ensure data is in expected format
  if(!inherits(x, "fG_format")) x <- format_data(x, ...)

  fit <- lapply(split(x, x$id),
                function(xx) {
                  prefilter(x = xx,
                            vmax = vmax,
                            ang = ang,
                            distlim = distlim,
                            spdf = spdf,
                            min.dt = min.dt,
                            emf = emf)
                })
  fit_result = try(do.call(rbind, fit))
  # extract x and y from geometry (POINT(x,y))
  fit_result$x = sapply(fit_result$geometry, function(x) x[1])
  fit_result$y = sapply(fit_result$geometry, function(x) x[2])

  cols_orig = c("lat", "lon")
  cols_pp = c("id", "date", "lc", "x", "y", "emf.x", "emf.y", "keep")
  df = cbind(fit_result[, cols_pp], x[, cols_orig])
  return(df)
}
