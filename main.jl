#!/usr/bin/env julia
include("greedy_naive.jl")
include("greedy.jl")
include("greedy_random.jl")
include("pilot.jl")
include("local_search.jl")
include("local_search_delta.jl")
include("vnd.jl")
include("vnd_delta.jl")
include("grasp.jl")
include("grasp_delta.jl")
include("gvns.jl")
include("gvns_delta.jl")
using CPUTime, JSON, Statistics
Plots.default(show=false)

function main()
    EXEC_TYPES = ["test", "competition"]
    EXEC_SIZES = ["50", "100", "200", "500", "1000", "2000", "5000", "10000"]
    BASE_DIR = "instances"
    STORE_DIR = "executions"
    N_VARIANCE = 3 # mandatory odd for median
    for (root, dirs, files) in walkdir(BASE_DIR)
        if files != [] # loop reached instance txts
            _, instance_size, instance_type = splitpath(root)
            if instance_type ∉ EXEC_TYPES || instance_size ∉ EXEC_SIZES
                continue
            end
            for instance_path in files
                # parse instance name
                instance_name = splitext(instance_path)[1]
                instance_id = split(instance_name, "_")[1]
                # define output directory structure
                path_instance = joinpath(pwd(), "{}", instance_size, instance_type)
                path_instance_id = joinpath(path_instance, instance_id)
                path_in = replace(path_instance, "{}" => BASE_DIR)
                path_out = replace(path_instance_id, "{}" => STORE_DIR)
                mkpath(path_out)
                # parse in PDP instance
                instance = read_PDPInstance(path_in, instance_path)
                # use deterministic greedy heuristic as baseline
                file_dg = joinpath(path_out, "greedy_deterministic", "config.json")
                score::Float64, solution::PDPSolutionVector = Inf, []
                if isfile(file_dg)
                    file_dict_dg = JSON.parsefile(file_dg)
                    score, solution = file_dict_dg["best_score"], file_dict_dg["best_solution"]
                else
                    _, _, solution, score = greedy_heuristic_one_extend(instance, verbose=true)
                end
                # define all algorithm runs
                algorithm_args_list = [
                    # non-delta algorithms
                    ("greedy_naive", greedy_heuristic_one_extend_pair, Dict(), false, 100),
                    ("pilot_g5", pilot_heuristic_full_rollout, Dict(:lookahead_gamma => 5), false, 100),
                    ("pilot_g25", pilot_heuristic_full_rollout, Dict(:lookahead_gamma => 25), false, 100),
                    # delta algorithms
                    ("greedy_deterministic", greedy_heuristic_one_extend, Dict(), false, 10000),
                    ("greedy_random", greedy_heuristic_one_extend_random, Dict(), true, 10000),
                    ("local_search_delta_in_random", delta_local_search, Dict(
                            :solution => solution,
                            :score => score,
                            :neighborhood_func => delta_neighbor_in_switch_location,
                            :step_func => delta_step_random), true, 10000),
                    ("local_search_delta_btw_random", delta_local_search, Dict(
                            :solution => solution,
                            :score => score,
                            :neighborhood_func => delta_neighbor_between_switch_request,
                            :step_func => delta_step_random), true, 10000),
                    ("local_search_delta_insq_random", delta_local_search, Dict(
                            :solution => solution,
                            :score => score,
                            :neighborhood_func => delta_neighbor_in_subsequence,
                            :step_func => delta_step_random), true, 10000),
                    ("local_search_delta_in_best", delta_local_search, Dict(
                            :solution => solution,
                            :score => score,
                            :neighborhood_func => delta_neighbor_in_switch_location,
                            :step_func => delta_step_best), true, 10000),
                    ("local_search_delta_btw_best", delta_local_search, Dict(
                            :solution => solution,
                            :score => score,
                            :neighborhood_func => delta_neighbor_between_switch_request,
                            :step_func => delta_step_best), true, 10000),
                    ("local_search_delta_insq_best", delta_local_search, Dict(
                            :solution => solution,
                            :score => score,
                            :neighborhood_func => delta_neighbor_in_subsequence,
                            :step_func => delta_step_best), true, 10000),
                    ("vnd_delta", delta_variable_neighborhood_descent, Dict(
                            :solution => solution,
                            :score => score), false, 10000),
                    ("grasp_delta", delta_GRASP, Dict(
                            :solution => solution,
                            :score => score), true, 10000),
                    ("gvns_delta", delta_general_variable_neighborhood_search, Dict(
                            :solution => solution,
                            :score => score), true, 10000),
                ]

                for (algo_idx, (algorithm_name, algorithm, args, is_random, size_cap)) in enumerate(algorithm_args_list)
                    # limit algorithms to realistic sizes
                    if parse(Int64, instance_size) > size_cap
                        continue
                    end
                    # only rerun if algorithm files do not exist
                    path_algo = joinpath(path_out, algorithm_name)
                    mkpath(path_algo)
                    file_json = joinpath(path_algo, "config.json")
                    file_txt = joinpath(path_algo, "submit.txt")
                    file_png = joinpath(path_algo, "visual.png")
                    avg_score, avg_solution = Inf, []
                    if isfile(file_json) # file exists
                        file_dict = JSON.parsefile(file_json)
                        avg_score, avg_solution = file_dict["best_score"], file_dict["best_solution"]
                    else # generate run file
                        @info "Executing run for $instance_id using $algorithm_name."
                        # rerun multiple times if random
                        seed_runs = []
                        n_runs = is_random ? N_VARIANCE : 1
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
                                "args" => args,
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

if abspath(PROGRAM_FILE) == @__FILE__ # if main
    main()
end