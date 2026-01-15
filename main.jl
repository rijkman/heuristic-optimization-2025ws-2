#!/usr/bin/env julia
include("./scripts/greedy.jl")
include("./scripts/greedy_random.jl")
include("./scripts/greedy_insertion.jl")
include("./scripts/gvns.jl")
include("./scripts/acs.jl")
include("./scripts/lns.jl")
Plots.default(show=false)

BASE_DIR = "data"
READ_DIR = "instances"

ArgDict = Dict{Symbol,Any}
@kwdef struct AlgorithmConfig{F<:Function}
    algorithm_name::String
    algorithm::F
    argset::ArgDict
    use_sol::Bool
    use_tune::Bool
    is_random::Bool
    instance_cap::Int64
end

function main_struct(store_dir::String,
    exec_types::Vector{String},
    exec_sizes::Vector{String},
    algorithm_names::Vector{String};
    instance_sep::Bool=true,
)
    for (root, _, files) in walkdir(joinpath(BASE_DIR, READ_DIR))
        if files != [] # loop reached instance txts
            _, _, instance_size, instance_type = splitpath(root)
            if instance_type ∉ exec_types || instance_size ∉ exec_sizes
                continue
            end
            # define output directory structure
            path_instance = joinpath(pwd(), BASE_DIR, "{}", instance_size, instance_type)
            path_instance_out = replace(path_instance, "{}" => store_dir)
            mkpath(path_instance_out)
            for name in algorithm_names
                # define output file structure
                if instance_sep
                    for instance_file in files
                        # parse instance name
                        instance_name = splitext(instance_file)[1]
                        instance_id = split(instance_name, "_")[1]
                        path_out = joinpath(path_instance_out, instance_id)
                        mkpath(path_out)
                        path_algo = joinpath(path_out, name)
                        mkpath(path_algo)
                    end
                else
                    path_algo = joinpath(path_instance_out, name)
                    mkpath(path_algo)
                end
            end
        end
    end
end

function main(
    store_dir::String,
    exec_types::Vector{String},
    exec_sizes::Vector{String},
    algorithm_configs::Vector{<:AlgorithmConfig};
    n_variance::Int64=3, # mandatory odd for median
)
    main_struct(store_dir, exec_types, exec_sizes, [cfg.algorithm_name for cfg in algorithm_configs])
    for (root, _, files) in walkdir(joinpath(BASE_DIR, READ_DIR))
        if files != [] # loop reached instance txts
            _, _, instance_size, instance_type = splitpath(root)
            if instance_type ∉ exec_types || instance_size ∉ exec_sizes
                continue
            end
            # define output directory structure
            path_base = joinpath(pwd(), BASE_DIR)
            path_instance = joinpath(path_base, "{}", instance_size, instance_type)
            path_instance_in = replace(path_instance, "{}" => READ_DIR)
            path_instance_out = replace(path_instance, "{}" => store_dir)
            path_instance_args = joinpath(path_base, "tuning", instance_size, "train")
            for instance_file in files
                # parse instance name
                instance_name = splitext(instance_file)[1]
                instance_id = split(instance_name, "_")[1]
                # define output file structure
                path_in = joinpath(path_instance_in, instance_file)
                path_out = joinpath(path_instance_out, instance_id)
                # parse in PDP instance
                instance = read_PDPInstance(path_in)
                # use deterministic greedy heuristic as baseline
                file_gd = joinpath(path_out, "greedy_deterministic", "config.json")
                score::Float64, solution::PDPSolutionVector = Inf, []
                if isfile(file_gd)
                    file_dict_dg = JSON.parsefile(file_gd)
                    score, solution = file_dict_dg["best_score"], file_dict_dg["best_solution"]
                else
                    _, _, solution, score = greedy_heuristic_one_extend(instance, verbose=true)
                end
                # run given argument configs
                for algo_config in algorithm_configs
                    (; algorithm_name, algorithm, argset, use_sol, use_tune, is_random, instance_cap) = algo_config
                    # limit algorithms to realistic sizes
                    if parse(Int64, instance_size) > instance_cap
                        continue
                    end
                    # only rerun if algorithm files do not exist
                    path_algo = joinpath(path_out, algorithm_name)
                    file_json = joinpath(path_algo, "config.json")
                    file_txt = joinpath(path_algo, "submit.txt")
                    file_png = joinpath(path_algo, "visual.png")
                    avg_score, avg_solution = Inf, []
                    if isfile(file_json) # file exists
                        file_dict = JSON.parsefile(file_json)
                        avg_score, avg_solution = file_dict["best_score"], file_dict["best_solution"]
                    else # generate run file
                        @info "Executing run for $instance_id using $algorithm_name."
                        # construct arguments list
                        args = argset
                        if use_sol # add initial solution
                            args_sol = Dict(:solution => solution, :score => score)
                            args = merge(args, args_sol)
                        end
                        if use_tune # use tuned parameters
                            path_args = joinpath(path_instance_args, algorithm_name, "irace.csv")
                            args_tune = Dict(pairs(first(CSV.File(path_args))))
                            args_algo = collect(Iterators.flatten(Base.kwarg_decl.(methods(algorithm))))
                            args_valid = Dict(k => v for (k, v) in args_tune if k in args_algo)
                            args = merge(args, args_valid)
                        end
                        # rerun multiple times if random
                        seed_runs = []
                        n_runs = is_random ? n_variance : 1
                        for seed in 1:n_runs
                            Random.seed!(seed)
                            # keep track of algorithm run configuration and results
                            CPUtic()
                            iter_score, iter_n, solution, score = algorithm(instance; args...)
                            runtime = CPUtoc()
                            seed_execution = (iter_score, iter_n, solution, score, runtime)
                            push!(seed_runs, seed_execution)
                        end
                        # calculate robust average of algorithm runs
                        median_run = filter(run -> run[4] == median(getindex.(seed_runs, 4)), seed_runs)[1]
                        avg_iter_score, avg_iter_n, avg_solution, avg_score, avg_runtime = median_run
                        # store run configuration as json
                        open(file_json, "w") do f
                            args_str = Dict(k => (v isa Function ? String(nameof(v)) : v) for (k, v) in args)
                            JSON.print(f, Dict(
                                "instance_size" => instance_size,
                                "instance_type" => instance_type,
                                "instance_id" => instance_id,
                                "algorithm" => algorithm_name,
                                "iter_score" => avg_iter_score,
                                "iter_n" => avg_iter_n,
                                "best_solution" => avg_solution,
                                "best_score" => avg_score,
                                "best_runtime" => avg_runtime,
                                "args" => args_str,
                            ))
                        end
                    end
                    if !isfile(file_txt)
                        # store solution submission as txt
                        open(file_txt, "w") do f
                            println(f, instance_name)
                            for route_k in avg_solution
                                println(f, join(route_k, " "))
                            end
                        end
                    end
                    if !isfile(file_png)
                        # visualize solution path as png
                        instance_plot = visualize(instance, avg_solution, avg_score)
                        savefig(instance_plot, file_png)
                    end
                end
            end
        end
    end
end

function main_init()
    # [task 1] - implementation is remaining scripts .jl

    # [task 2] - fairness exploration
    FAIRNESS_STORE_DIR = "fairness"
    FAIRNESS_EXEC_TYPES = ["train"]
    FAIRNESS_EXEC_SIZES = ["100"] # ["100", "1000"]
    FAIRNESS_ALGORITHMS = [
        AlgorithmConfig(
            algorithm_name="greedy_jain",
            algorithm=greedy_heuristic_one_extend,
            argset=ArgDict(:fairness => jain_fairness),
            use_sol=false,
            use_tune=false,
            is_random=false,
            instance_cap=10000
        ),
        AlgorithmConfig(
            algorithm_name="greedy_maxmin",
            algorithm=greedy_heuristic_one_extend,
            argset=ArgDict(:fairness => max_min_fairness),
            use_sol=false,
            use_tune=false,
            is_random=false,
            instance_cap=10000
        ),
        AlgorithmConfig(
            algorithm_name="greedy_gini",
            algorithm=greedy_heuristic_one_extend,
            argset=ArgDict(:fairness => gini_fairness),
            use_sol=false,
            use_tune=false,
            is_random=false,
            instance_cap=10000
        ),
    ]
    main(FAIRNESS_STORE_DIR, FAIRNESS_EXEC_TYPES, FAIRNESS_EXEC_SIZES, FAIRNESS_ALGORITHMS)

    # [task 3] - parameter tuning using [irace]
    TUNING_STORE_DIR = "tuning"
    TUNING_EXEC_TYPES = ["train"]
    TUNING_EXEC_SIZES = ["50", "100", "200", "500", "1000"]
    TUNING_ALGORITHMS = ["acs", "lns"]
    main_struct(TUNING_STORE_DIR, TUNING_EXEC_TYPES, TUNING_EXEC_SIZES, TUNING_ALGORITHMS; instance_sep=false)

    # [task 4] - best experiments; using tuning results of [task 3]
    EXP_STORE_DIR = "experiments"
    EXP_EXEC_TYPES = ["test"]
    EXP_EXEC_SIZES = ["50"] # ["50", "100", "200", "500", "1000", "2000"]
    EXP_ALGORITHMS = [
        # AlgorithmConfig(
        #     algorithm_name="gvns",
        #     algorithm=general_variable_neighborhood_search,
        #     argset=ArgDict(),
        #     use_sol=true,
        #     use_tune=false,
        #     is_random=true,
        #     instance_cap=10000
        # ),
        # AlgorithmConfig(
        #     algorithm_name="acs",
        #     algorithm=ant_colony_system,
        #     argset=ArgDict(),
        #     use_sol=true,
        #     use_tune=true,
        #     is_random=true,
        #     instance_cap=10000
        # ),
        AlgorithmConfig(
            algorithm_name="lns",
            algorithm=large_neighborhood_search,
            argset=ArgDict(),
            use_sol=true,
            use_tune=true,
            is_random=true,
            instance_cap=10000
        ),
    ]
    main(EXP_STORE_DIR, EXP_EXEC_TYPES, EXP_EXEC_SIZES, EXP_ALGORITHMS)

    # [task 5] - significance tests; using results of [task 4]

    # [task X] - competition
    COMP_STORE_DIR = "competition"
    COMP_EXEC_TYPES = ["competition"]
    COMP_EXEC_SIZES = ["50", "100", "200", "500", "1000", "2000", "5000", "10000"]
    COMP_ALGORITHMS = [
        AlgorithmConfig(
            algorithm_name="acs",
            algorithm=ant_colony_system,
            argset=ArgDict(),
            use_sol=true,
            use_tune=true,
            is_random=false,
            instance_cap=10000
        ),
        AlgorithmConfig(
            algorithm_name="lns",
            algorithm=large_neighborhood_search,
            argset=ArgDict(),
            use_sol=true,
            use_tune=true,
            is_random=false,
            instance_cap=10000
        ),
    ]
    main(COMP_STORE_DIR, COMP_EXEC_TYPES, COMP_EXEC_SIZES, COMP_ALGORITHMS)
end

if abspath(PROGRAM_FILE) == @__FILE__ # if main
    main_init()
end