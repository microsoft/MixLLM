import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['VLLM_USE_BATCHLLM'] = '1'

import random
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time
import datetime
from pandas import read_table

def prepare_benchmark_tokens(tokenizer,context_len, prompt_len):
    context = """
    A river delta is a landform shaped like a triangle, created by the deposition of sediment that is carried by a river and enters slower-moving or stagnant water. This occurs at a river mouth, when it enters an ocean, sea, estuary, lake, reservoir, or (more rarely) another river that cannot carry away the supplied sediment. It is so named because its triangle shape resembles the uppercase Greek letter delta, Δ. The size and shape of a delta are controlled by the balance between watershed processes that supply sediment, and receiving basin processes that redistribute, sequester, and export that sediment. The size, geometry, and location of the receiving basin also plays an important role in delta evolution.

    River deltas are important in human civilization, as they are major agricultural production centers and population centers. They can provide coastline defense and can impact drinking water supply. They are also ecologically important, with different species' assemblages depending on their landscape position. On geologic timescales, they are also important carbon sinks.

    Etymology
    A river delta is so named because the shape of the Nile Delta approximates the triangular uppercase Greek letter delta. The triangular shape of the Nile Delta was known to audiences of classical Athenian drama; the tragedy Prometheus Bound by Aeschylus refers to it as the "triangular Nilotic land", though not as a "delta". Herodotus's description of Egypt in his Histories mentions the Delta fourteen times, as "the Delta, as it is called by the Ionians", including describing the outflow of silt into the sea and the convexly curved seaward side of the triangle. Despite making comparisons to other river-systems' deltas, Herodotus did not describe them as "deltas". The Greek historian Polybius likened the land between the Rhône and Isère rivers to the Nile Delta, referring to both as islands, but did not apply the word delta. According to the Greek geographer Strabo, the Cynic philosopher Onesicritus of Astypalaea, who accompanied Alexander the Great's conquests in India, reported that Patalene (the delta of the Indus River) was "a delta" (Koinē Greek: καλεῖ δὲ τὴν νῆσον δέλτα, romanized: kalei de tēn nēson délta, lit. 'he calls the island a delta'). The Roman author Arrian's Indica states that "the delta of the land of the Indians is made by the Indus river no less than is the case with that of Egypt".

    As a generic term for the landform at the mouth of river, the word delta is first attested in the English-speaking world in the late 18th century, in the work of Edward Gibbon.

    Formation
    River deltas form when a river carrying sediment reaches a body of water, such as a lake, ocean, or a reservoir. When the flow enters the standing water, it is no longer confined to its channel and expands in width. This flow expansion results in a decrease in the flow velocity, which diminishes the ability of the flow to transport sediment. As a result, sediment drops out of the flow and is deposited as alluvium, which builds up to form the river delta. Over time, this single channel builds a deltaic lobe (such as the bird's-foot of the Mississippi or Ural river deltas), pushing its mouth into the standing water. As the deltaic lobe advances, the gradient of the river channel becomes lower because the river channel is longer but has the same change in elevation (see slope).

    As the gradient of the river channel decreases, the amount of shear stress on the bed decreases, which results in the deposition of sediment within the channel and a rise in the channel bed relative to the floodplain. This destabilizes the river channel. If the river breaches its natural levees (such as during a flood), it spills out into a new course with a shorter route to the ocean, thereby obtaining a steeper, more stable gradient. Typically, when the river switches channels in this manner, some of its flow remains in the abandoned channel. Repeated channel-switching events build up a mature delta with a distributary network.

    Another way these distributary networks form is from the deposition of mouth bars (mid-channel sand and/or gravel bars at the mouth of a river). When this mid-channel bar is deposited at the mouth of a river, the flow is routed around it. This results in additional deposition on the upstream end of the mouth-bar, which splits the river into two distributary channels. A good example of the result of this process is the Wax Lake Delta.

    In both of these cases, depositional processes force redistribution of deposition from areas of high deposition to areas of low deposition. This results in the smoothing of the planform (or map-view) shape of the delta as the channels move across its surface and deposit sediment. Because the sediment is laid down in this fashion, the shape of these deltas approximates a fan. The more often the flow changes course, the shape develops as closer to an ideal fan, because more rapid changes in channel position result in more uniform deposition of sediment on the delta front. The Mississippi and Ural River deltas, with their bird's-feet, are examples of rivers that do not avulse often enough to form a symmetrical fan shape. Alluvial fan deltas, as seen by their name, avulse frequently and more closely approximate an ideal fan shape.

    Most large river deltas discharge to intra-cratonic basins on the trailing edges of passive margins due to the majority of large rivers such as the Mississippi, Nile, Amazon, Ganges, Indus, Yangtze, and Yellow River discharging along passive continental margins. This phenomenon is due mainly to three factors: topography, basin area, and basin elevation. Topography along passive margins tend to be more gradual and widespread over a greater area enabling sediment to pile up and accumulate over time to form large river deltas. Topography along active margins tend to be steeper and less widespread, which results in sediments not having the ability to pile up and accumulate due to the sediment traveling into a steep subduction trench rather than a shallow continental shelf.

    There are many other lesser factors that could explain why the majority of river deltas form along passive margins rather than active margins. Along active margins, orogenic sequences cause tectonic activity to form over-steepened slopes, brecciated rocks, and volcanic activity resulting in delta formation to exist closer to the sediment source. When sediment does not travel far from the source, sediments that build up are coarser grained and more loosely consolidated, therefore making delta formation more difficult. Tectonic activity on active margins causes the formation of river deltas to form closer to the sediment source which may affect channel avulsion, delta lobe switching, and auto cyclicity. Active margin river deltas tend to be much smaller and less abundant but may transport similar amounts of sediment. However, the sediment is never piled up in thick sequences due to the sediment traveling and depositing in deep subduction trenches.

    Types
    Deltas are typically classified according to the main control on deposition, which is a combination of river, wave, and tidal processes, depending on the strength of each. The other two factors that play a major role are landscape position and the grain size distribution of the source sediment entering the delta from the river.

    Fluvial-dominated deltas
    Fluvial-dominated deltas are found in areas of low tidal range and low wave energy. Where the river water is nearly equal in density to the basin water, the delta is characterized by homopycnal flow, in which the river water rapidly mixes with basin water and abruptly dumps most of its sediment load. Where the river water has higher density than basin water, typically from a heavy load of sediment, the delta is characterized by hyperpycnal flow in which the river water hugs the basin bottom as a density current that deposits its sediments as turbidites. When the river water is less dense than the basin water, as is typical of river deltas on an ocean coastline, the delta is characterized by hypopycnal flow in which the river water is slow to mix with the denser basin water and spreads out as a surface fan. This allows fine sediments to be carried a considerable distance before settling out of suspension. Beds in a hypocynal delta dip at a very shallow angle, around 1 degree.

    Fluvial-dominated deltas are further distinguished by the relative importance of the inertia of rapidly flowing water, the importance of turbulent bed friction beyond the river mouth, and buoyancy. Outflow dominated by inertia tend to form Gilbert type deltas. Outflow dominated by turbulent friction is prone to channel bifurcation, while buoyancy-dominated outflow produces long distributaries with narrow subaqueous natural levees and few channel bifurcations.

    The modern Mississippi River delta is a good example of a fluvial-dominated delta whose outflow is buoyancy-dominated. Channel abandonment has been frequent, with seven distinct channels active over the last 5000 years. Other fluvial-dominated deltas include the Mackenzie delta and the Alta delta.

    Gilbert deltas
    A Gilbert delta (named after Grove Karl Gilbert) is a type of fluvial-dominated delta formed from coarse sediments, as opposed to gently-sloping muddy deltas such as that of the Mississippi. For example, a mountain river depositing sediment into a freshwater lake would form this kind of delta.  It is commonly a result of homopycnal flow. Such deltas are characterized by a tripartite structure of topset, foreset, and bottomset beds. River water entering the lake rapidly deposits its coarser sediments on the submerged face of the delta, forming steeping dipping foreset beds. The finer sediments are deposited on the lake bottom beyond this steep slope as more gently dipping bottomset beds. Behind the delta front, braided channels deposit the gently dipping beds of the topset on the delta plain.

    While some authors describe both lacustrine and marine locations of Gilbert deltas, others note that their formation is more characteristic of the freshwater lakes, where it is easier for the river water to mix with the lakewater faster (as opposed to the case of a river falling into the sea or a salt lake, where less dense fresh water brought by the river stays on top longer). Gilbert himself first described this type of delta on Lake Bonneville in 1885. Elsewhere, similar structures occur, for example, at the mouths of several creeks that flow into Okanagan Lake in British Columbia and forming prominent peninsulas at Naramata, Summerland, and Peachland.

    Wave-dominated deltas
    In wave dominated deltas, wave-driven sediment transport controls the shape of the delta, and much of the sediment emanating from the river mouth is deflected along the coast line. The relationship between waves and river deltas is quite variable and largely influenced by the deepwater wave regimes of the receiving basin. With a high wave energy near shore and a steeper slope offshore, waves will make river deltas smoother. Waves can also be responsible for carrying sediments away from the river delta, causing the delta to retreat. For deltas that form further upriver in an estuary, there are complex yet quantifiable linkages between winds, tides, river discharge, and delta water levels.

    """

    prompt = """
    Objective: The purpose of this task is to generate a concise and coherent summary of the provided document, capturing all the critical points and themes without any loss of essential information. The summary should be comprehensive enough to serve as a standalone overview of the document's content.

    -Read through the entire document carefully to understand the context, arguments, and data presented.
    -Identify the main points and arguments in each section of the document..

    Expected Outcome:
    The final summary should provide a clear and accurate representation of the document's content, allowing readers to quickly grasp the essence of the full report without needing to -read it in its entirety. The summary should be no longer than 10% of the original document's length, ensuring it remains digestible and focused. Given the provided document, please give me the final output concise and coherent summary in detail:

    """
    common = tokenizer.encode(context,add_special_tokens=False,return_tensors='pt')[0,0:context_len].tolist()
    dist = tokenizer.encode(prompt,add_special_tokens=False,return_tensors='pt')[0,0:prompt_len].tolist()

    return common, dist

if __name__=='__main__':
    if torch.cuda.get_device_properties(0).name[:3] == 'AMD':
        device = 'amd'
        torch_version = torch.version.hip
    elif torch.cuda.get_device_properties(0).name[:6] == 'NVIDIA':
        device = 'nvidia'
        torch_version = torch.version.cuda
    output_file = '/data/test_outputs/test_context_share_grid.txt'
    f = open(output_file, 'a')
    print(f'////////////////////////////',file = f)
    print(f' {datetime.datetime.now()} vllm performace test',file = f)
    print(f'{device},{torch_version}',file = f)
    engine_path = '/data/qwen_2_5_7b'
    prompt_num = 6400
    share_degree_list = [16]
    token_batching_degree_list = [100]
    enable_clustering_list = [True, False]
    disable_cs_kernels_list = [0]
    context_len = 2000
    prompt_len = 200
    vllm_share = False
    enable_chunked_prefill = True
    max_num_batched_tokens = 2048
    tokenizer = AutoTokenizer.from_pretrained(engine_path)
    benchmark_common, benchmark_dist = prepare_benchmark_tokens(tokenizer,context_len,prompt_len)

    print(f'engine_path:{engine_path},enable_chunked:{enable_chunked_prefill},max_num_batched_tokens:{max_num_batched_tokens}',file = f)
    header = f'{"tbd":>6} {"sd":>6} {"cluster":>8} {"cs_kern":>8} {"time":>8} {"throughput":>12}'
    print(header, file=f)
    print(header)

    for enable_clustering in enable_clustering_list:
        for disable_cs in disable_cs_kernels_list:
            # Skip invalid/redundant combos
            # if enable_clustering and disable_cs:
            #     # clustering ON + CS kernel OFF: scheduler groups for CS but kernel skipped — crashes
            #     continue
            # if not enable_clustering and not disable_cs:
            #     # clustering OFF + CS kernel ON: no grouping so CS kernel never triggers — same as OFF
            #     continue
            for token_batching_degree in token_batching_degree_list:
                label = f'cluster={enable_clustering}, cs_kern={"OFF" if disable_cs else "ON"}, tbd={token_batching_degree}'
                print(f'\n===== {label} =====')
                print(f'\n===== {label} =====', file=f)

                os.environ["DISABLE_CS_KERNELS"] = str(disable_cs)

                vllm_engine = LLM(model=engine_path,
                           enable_chunked_prefill=enable_chunked_prefill,
                           max_num_batched_tokens=max_num_batched_tokens,
                           token_batching_degree=token_batching_degree,
                           tensor_parallel_size=1,
                           disable_sliding_window=True,
                           block_size=64,
                           dtype="float16",
                           enforce_eager=True,
                           enable_ahead_of_prefix_clustering=enable_clustering,
                           max_model_len=3000,
                           )
                sampling_params = SamplingParams(temperature=0.01,
                                                 top_p=0.1,
                                                 max_tokens=100,
                                                 ignore_eos=True
                                                 )

                for share_degree in share_degree_list:
                    benchmark_input = [[benchmark_common,[benchmark_dist]*share_degree]]*(prompt_num//share_degree)
                    random.seed(42)
                    random.shuffle(benchmark_input)
                    # Flatten for non-clustering mode: CSGroup format not supported
                    if not enable_clustering:
                        flat_input = []
                        for group in benchmark_input:
                            common = group[0]
                            for dist in group[1]:
                                flat_input.append(common + dist)
                        random.seed(42)
                        random.shuffle(flat_input)
                        gen_input = flat_input
                    else:
                        gen_input = benchmark_input
                    print(f'  Running tbd={token_batching_degree}, sd={share_degree}, cluster={enable_clustering}, cs_kern={"OFF" if disable_cs else "ON"}: {prompt_num} prompts')
                    t1 = time.time()
                    output = vllm_engine.generate(prompt_token_ids=gen_input,
                                           sampling_params=sampling_params
                                           )
                    t2 = time.time()
                    line = f'{token_batching_degree:6} {share_degree:6} {str(enable_clustering):>8} {"OFF" if disable_cs else "ON":>8} {t2-t1:8.2f} {prompt_num/(t2-t1):12.2f}'
                    print(f'  {line}')
                    print(line, file=f)
                    f.flush()

                del vllm_engine
                import gc; gc.collect()
                torch.cuda.empty_cache()

    f.close()
    print(f'\nResults saved to {output_file}')
