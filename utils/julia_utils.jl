using CodecBzip2
using DataFrames
using DataStructures
using CSV
using PyCall
using Dates
using Arrow
using ProgressBars
using GRIB
import Pandas

pd = pyimport("pandas")
np = pyimport("numpy")
pickle = pyimport("pickle")
lzma = pyimport("lzma")

mainDir = dirname(dirname(abspath(@__FILE__)))
fullpath = path -> joinpath(mainDir, path)

##############################################################################################################################
"""                                            I. Compression / Decompression                                              """
##############################################################################################################################

@doc """Stores df compressed columns in the specified folder. Slower but higher compression than Julia's compression funcs."""
function compress_df(df, folder)
    if ! isdir(folder)
        mkdir(folder)
    end
    for col in names(df)
        array = getproperty(df, col)
        array_str = string(chop(string(array), head=1, tail=1))
        str_compresed = transcode(Bzip2Compressor, array_str)
        write(joinpath(folder, "$col.bz2"), str_compresed)
    end
end

@doc """
Decompress df, stored using compress_df. Slower but higher compression than Julia's compression funcs.
All columns are formated to substring arrays.
"""
function decompress_df(folder)
    data = Dict()
    for file in readdir(folder)
        column = split(file, ".")[1]
        compressed = read(joinpath(folder, file))
        plain = transcode(Bzip2Decompressor, compressed)
        array = split(String(plain), ", ")
        data[column] = array
    end
    df = DataFrame(data)
end


##############################################################################################################################
"""                                                 II. Processing python files                                            """
##############################################################################################################################

function load_python_lzma(path)
    pickle.load(lzma.LZMAFile(path, "rb"))
end

@doc """Returns Julia's DataFrame of a Python Pandas DataFrame. All the columns are formatted to strings."""
function pd_to_DataFrame(df_pd, cols=nothing)
    if isnothing(cols)
        cols = [string(c) for c in df_pd.columns]
    end
    data = [getproperty(df_pd, col).values for col in cols]
    data_parsed = Dict( (col, convert(Array{String,1}, d)) for (col, d) in zip(cols, data) )
    df_jl = DataFrame(data_parsed)
end

@doc """Returns Python Pandas DataFrame of a Julia's DataFrame. All the columns are formatted to strings."""
function DataFrame_to_pd(df_jl, cols=nothing)
    if isnothing(cols)
        cols = [string(c) for c in df_pd.columns]
    end
    data = [getproperty(df_jl, col).values for col in cols]
    data_parsed = Dict( (col, convert(Array{String,1}, d)) for (col, d) in zip(cols, data) )
    df_pd = pd.DataFrame(Pandas.DataFrame(data_parsed))
end


##############################################################################################################################
"""                                                   III. String processing                                               """
##############################################################################################################################

function str_to_date(x, date_format=dateformat"yyyy-mm-dd HH:MM:SS")
    y = [DateTime(d, date_format) for d in x]
end


##############################################################################################################################
"""                                                   III. GRIB Processing                                                 """
##############################################################################################################################

function closest_index(x, val)
    ibest = 1
    dxbest = abs(x[ibest]-val)
    for I in eachindex(x)
        dx = abs(x[I]-val)
        if dx < dxbest
            dxbest = dx
            ibest = I
        end
    end
    ibest
end

function closest_indices(x, val)
    idxs = []
    ibest = 1
    dxbest = abs(x[ibest]-val)
    for I in eachindex(x)
        dx = abs(x[I]-val)
        if dx < dxbest
            idxs = [I]
            dxbest = dx
            ibest = I
        elseif dx == dxbest
            push!(idxs, I)
        end
    end
    idxs
end

function value_in_message(m, lat, lon)
    lons, lats, magnitude = [vec(i) for i in data(m)]
    lat_idxs = closest_indices(lats, lat)
    lon_idx = closest_index(lons[lat_idxs], lon)
    idx = lat_idxs[1] + lon_idx - 1
    magnitude[idx]
end

function retrieve_df_year(year; positive_lon=false, v2=false)
    if v2
        df_path = fullpath("utils/data/mobility_data_v2_jl.zstd")
    else
        df_path = fullpath("utils/data/mobility_data_jl.zstd")
    end
    df = DataFrames.DataFrame(Arrow.Table(df_path))
    df = df[Dates.Year.(df.DATE_TIME) .== Dates.Year(year), :]
    if positive_lon
        df.LONGITUDE .+= 180
    end
    sort(df, :DATE_TIME)
end

function get_var_dict(f)
    seekstart(f)
    weather_data = Dict()
    var = Message(f)["name"]
    while isnothing(get(weather_data, var, nothing))
        weather_data[var] = []
        var = Message(f)["name"]
    end
    seekstart(f)
    weather_data
end


function get_col_type(gribfile)
    function aux(f)
        hours = Set()
        for i in 1:10
            push!(hours, Message(f)["hour"])
        end
        if length(hours) > 2
            return "analysis"
        else
            if length(split(Message(f)["stepRange"], "-")) > 1
                return "forecast range"
            else
                return "forecast instant"
            end
        end
    end

    if isa(gribfile, GribFile)
        return aux(gribfile)
    else
        GribFile(gribfile) do f
            return aux(f)
        end
    end
end

function get_compute_hour(col_type)

    compute_hour = if col_type == "forecast range"
                       message -> (message["hour"] + parse(Int64, split(message["stepRange"], "-")[2])) % 24
                   elseif col_type == "forecast instant"
                       message -> (message["hour"] + parse(Int64, message["stepRange"])) % 24
                   else
                       message -> message["hour"]
                   end
end

function get_compute_date(col_type)

    get_step = if col_type == "forecast range"
                    m -> parse(Int64, split(m["stepRange"], "-")[2])
               elseif col_type == "forecast instant"
                    m -> parse(Int64, m["stepRange"])
               else
                    nothing
               end

    compute_date = if occursin("forecast", col_type)
                       function(message)
                           real_date = if message["hour"] == 6
                                           message["date"]
                                       else
                                           step = get_step(message)
                                           if step > 5
                                               day = message["day"]
                                               if day > 27
                                                   d = Dates.Date(message["year"], message["month"], message["day"]) + Dates.Day(1) # there can be a change in the month
                                                   Dates.year(d) * 10000 + Dates.month(d) * 100 + Dates.day(d)
                                               else
                                                   message["date"] + 1
                                               end
                                           else
                                               message["date"]
                                           end
                                       end
                       end
                   else
                       message -> message["date"]
                   end
end

function is_col_in_df(path, df)
    GribFile(path) do f
        weather_data = get_var_dict(f)
        key = collect(keys(weather_data))[1]

        return any([key == col for col in names(df)])
    end
end

function get_weather_df(year; v2=false)
    if v2
        df_path = fullpath("utils/data/weather/v2/full_df_$year.zstd")
    else
        df_path = fullpath("utils/data/weather/full_df_$year.zstd")
    end
    df = if isfile(df_path)
             DataFrames.DataFrame(Arrow.Table(df_path))
         else
             retrieve_df_year(year, v2=v2)
         end
end

@doc """Concatenate weather dataframes for all years in the range [1985, 2019]"""
function full_weather_df(; v2=false)
    df = get_weather_df(1985, v2=v2)
    for year in 1986:2019
        df = vcat(df, get_weather_df(year, v2=v2))
    end
    df
end

@doc """Saves full weather dataframe to a csv file"""
function save_full_weather_df(; v2=false)
    df = full_weather_df(v2=v2)
    if v2
        df_path = fullpath("utils/data/weather/v2/full_df.csv")
    else
        df_path = fullpath("utils/data/weather/full_df.csv")
    end
    CSV.write(df_path, df)
end

@doc """The need of functions compute_date, compute_hour comes from the difference in the date-time specification in ERA5 variables:
        << The date and time of the data is specified with three MARS keywords: date, time and (forecast) step.
        For analyses, step=0 hours so that date and time specify the analysis date/time.
        For forecasts, date and time specify the forecast start time and step specifies the number of hours since that start time.>>"""
function process_gribfile(path, df)

    GribFile(path) do f
        global weather_data = get_var_dict(f)
        key = collect(keys(weather_data))[1]

        if any([key == col for col in names(df)])
            println("Column $key was already added.")
            weather_data = Dict()
        else
            int_parser = x -> ifelse(x < 10, "0$x", string(x))
            threshold_func = H -> ifelse(H > 23, 1, 0.5)
            col_type = get_col_type(f)
            println("Column $key type: $col_type")
            compute_date = get_compute_date(col_type)
            compute_hour = get_compute_hour(col_type)

            seekstart(f)
            message = Message(f)
            for row in Tables.rows(Tables.rowtable(df))
                date, lat, lon = row
                Y, M, D = Dates.year(date), Dates.month(date), Dates.day(date)
                pM, pD = int_parser(M), int_parser(D)
                H = Dates.hour(date) + Dates.minute(date)/60
                date = parse(Int, "$Y$pM$pD")

                threshold = threshold_func(H)

                while compute_date(message) < date
                    message = Message(f)
                end

                while abs(compute_hour(message) - H) > threshold
                    message = Message(f)
                end
                push!(weather_data[message["name"]], value_in_message(message, lat, lon))
            end
        end
    end
    weather_data
end

function complete_df(year; v2=false, saveDir=fullpath("utils/data/weather"), gribRoot=fullpath("utils/data/SingleLevels"))
    if v2
        saveDir = joinpath(saveDir, "v2")
        gribRoot = joinpath(gribRoot, "v2")
    end
    if !isdir(saveDir)
        mkdir(saveDir)
    end

    saving_path = joinpath(saveDir, "full_df_$year.zstd")
    if isfile(saving_path)
        df = DataFrames.DataFrame(Arrow.Table(saving_path))
    else
        df =  retrieve_df_year(year, v2=v2)
    end

    gribDir = joinpath(gribRoot, filter(folder -> occursin(string(year), folder), readdir(gribRoot))[1])
    gribfiles = filter(file -> occursin(string(year), file), readdir(gribDir))
    gribfiles = filter(file -> !occursin("month", file), gribfiles)
    breakline = "\n" * repeat("-", 100)  * "\n"

    weather_data_total = Dict()
    for gribfile in ProgressBar(gribfiles)
        println("Processing $gribfile")

        col_data = process_gribfile(joinpath(gribDir, gribfile), df)
        merge!(weather_data_total, col_data)
        println(breakline)
    end

    println("Data collection completed. Saving...")

    df_full = hcat(df, DataFrame(weather_data_total))
    try
        test_path = replace(saving_path, ".zstd" => "_test.zstd")
        Arrow.write(test_path, df_full, compress=:zstd)
        try
            mv(test_path, saving_path, force=true)
        catch
            println("Error while replacing savefile -> testfile")
        end
    catch
        println("Error while saving testfile")
    end


end
