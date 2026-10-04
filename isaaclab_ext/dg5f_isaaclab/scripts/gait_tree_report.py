"""Render only measured gait-tree results; never infer success from reward."""
import json,statistics,subprocess,shlex
from collections import Counter,defaultdict
from pathlib import Path
from gait_multiseed import paired_bootstrap
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'logs/gait_tree'
KEYS=['gait/successful_gait_cycles_per_episode','gait/successful_gait_cycle_rate','gait/meaningful_events_per_episode','gait/recovery_to_3plus_rate','gait/contact_switch_distance_p90','gait/goals_with_meaningful_switch_fraction','gait/two_tip_support_duration','gait/distinct_masks_per_episode','gait/distinct_fingers_switched_per_episode','goals_completed_per_episode','target_completion_rate','drop_rate','support_qualified_goals_per_episode','raw_mu','saturation','action_std']

def main():
    state=json.loads((OUT/'night_state.json').read_text());evaluations=state['evaluations'];groups=defaultdict(list)
    for label,row in evaluations.items():groups[label.rsplit('_s',1)[0]].append((label,row))
    # Duration-normalized event rates separate additional survival from more frequent gait.
    for pairs in groups.values():
        for label,row in pairs:
            record=next(iter(json.loads((OUT/(label+'.json')).read_text()).values()))
            row['episode_length_s']=record['episode_length_s']
            row['cycles_per_second']=row['gait/successful_gait_cycles_per_episode']/row['episode_length_s']
            row['goals_per_second']=row['goals_completed_per_episode']/row['episode_length_s']
    summaries={}
    for name,pairs in groups.items():
        rows=[x[1] for x in pairs];summary={k:statistics.mean(r[k] for r in rows) for k in KEYS+["episode_length_s","cycles_per_second","goals_per_second"] if all(isinstance(r.get(k),(float,int)) for r in rows)}
        summary.update(seeds=[r['seed'] for r in rows],checkpoint=rows[0]['checkpoint'])
        summary['tip_fraction_1']=statistics.mean(r['tip_count_fractions']['1'] for r in rows)
        summary['tip_fraction_2']=statistics.mean(r['tip_count_fractions']['2'] for r in rows)
        summary['tip_fraction_3plus']=statistics.mean(sum(r['tip_count_fractions'][str(k)] for k in (3,4,5)) for r in rows)
        summaries[name]=summary
    selected=state.get('selected_candidate');best=state.get('best_short');bestname='FINAL_LONG' if 'FINAL_LONG' in summaries else best+'_300' if best else 'CONTROL_300'
    mechanisms={'H2_GATE':'reward semantics','H3_CONTACT_MASK':'contact observability','H2_H3':'combination: reward semantics + contact observability','H4_GRASP_MECHANICS':'grasp-mechanics awareness','H4_H2':'combination: grasp mechanics + reward semantics','H5_CONTACT_MASK_TIMERS':'temporal organization'}
    conclusion=mechanisms.get(selected,'none of the tested hypotheses')
    # A selected smoke is evidence to investigate, not proof of organized gait.
    gait='PARTIAL' if selected else 'NO'
    final_pairs={}
    if 'FINAL_LONG' in groups:
        final={r['seed']:r for _,r in groups['FINAL_LONG']}
        for reference in ['ROOT','CONTROL_300',selected+'_300']:
            before={r['seed']:r for _,r in groups[reference]}
            seeds=sorted(final.keys() & before.keys())
            final_pairs[reference]={k:paired_bootstrap([final[s][k]-before[s][k] for s in seeds],draws=4000) for k in KEYS+['cycles_per_second','episode_length_s'] if all(k in final[s] and k in before[s] for s in seeds)}
    comparisons={p.stem:json.loads(p.read_text()) for p in OUT.glob('*_comparison.json')}
    diagnostics={}
    for name in dict.fromkeys(['ROOT','CONTROL_300',*(['H4_GRASP_MECHANICS_300'] if 'H4_GRASP_MECHANICS_300' in groups else []),bestname]):
        graph=Counter();loops=Counter();occupancy=defaultdict(list);per=defaultdict(Counter);filters=Counter();qdot=[];used=[];loo=defaultdict(list)
        for label,row in groups.get(name,[]):
            record=next(iter(json.loads((OUT/(label+'.json')).read_text()).values()));g=record['gait'];used.append(row['seed'])
            for f,v in g.get('leave_one_out_quality_at_release',{}).items():
                if v.get('count'):loo[f].append(v['mean'])
            graph.update(g['contact_graph']);loops.update(g.get('contact_mask_return_loops',{}));filters.update(g.get('event_filters',{}))
            for mask,v in g.get('contact_mask_occupancy',{}).items():occupancy[mask].append(v)
            for finger,v in g['per_finger'].items():
                for k in ('events','release_count','meaningful','successful_cycles'):per[finger][k]+=v.get(k,0)
                per[finger]['support_frequency_sum']+=v['support_frequency']
                per[finger]['displacement_mean_sum']+=v['displacement'].get('mean',0)
            qdot.append({k:record['smoothing'][k] for k in ('qdot_physx','qdot_fd','qdot_fd_after_1s')})
        diagnostics[name]={'seeds':used,'mask_occupancy':{k:statistics.mean(v) for k,v in occupancy.items()},'top_transitions':graph.most_common(20),'top_return_loops':loops.most_common(20),'per_finger':dict(per),'event_filters':dict(filters),'qdot_per_seed':qdot,'mean_leave_one_out_at_release_by_finger':{f:statistics.mean(v) for f,v in loo.items()}}
    result={'root':state['root_checkpoint'],'initial_head':state['initial_head'],'final_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'selected_candidate':selected,'best_short':best,'best_long':state.get('best_long_checkpoint'),'bottleneck_supported':conclusion,'true_gait_emerged':gait,'summaries':summaries,'comparisons':comparisons,'diagnostics':diagnostics,'long_curve':state.get('long_curve',[]),'videos':[str(p.relative_to(OUT)) for p in (OUT/'videos').glob('*.mp4')] if (OUT/'videos').exists() else [],'limitations':['One training seed (42); evaluation seeds measure evaluation variability, not training replication.','The full8-seed panel includes the3 smoke seeds:5 additional evaluation seeds, not8 wholly independent replications. Multiple candidate selection can inflate apparent improvements.','A negative 300-update smoke does not prove the mechanism can never help under a longer or different training regime.','Simulation states and the existing batch grasp-quality EMA restart at process boundaries; model/Adam/normalizers/iteration/LR are restored, but the old checkpoint never contained the environment EMA.','Seed pairing is approximate: policy-dependent episode resets and PhysX nondeterminism change trajectories.','Long run has no matched long control; smoke causality and descriptive learning curve must be distinguished.','Strict cycle requires a completed goal within 0.5s of recontact; slow valid manipulation can be missed.','Contact-mask return loops alone do not imply jitter: geometric displacement must be checked.','Cube-local contact centroid can change when contact patches change; threshold remains frozen from baseline.','Grasp Gramian assumes isotropic point-force bases and fingertip origins: a proxy, not friction-cone force closure.']}
    result['selection_review']=state.get('selection_review')
    result['final_paired_descriptive_comparisons']=final_pairs
    result['bottleneck_established']=False
    result['gait_verdict_basis']='Rare strict cycles increase, but an organized repeated gait distribution is not established; selected smoke evidence alone is not emergence.'
    if (OUT/'DELIVERED_ACTION_AUDIT.json').exists():result['delivered_action_audit']=json.loads((OUT/'DELIVERED_ACTION_AUDIT.json').read_text())
    if (OUT/'H4_TIME_NORMALIZED_DIAGNOSTIC.json').exists():result['H4_time_normalized_diagnostic']=json.loads((OUT/'H4_TIME_NORMALIZED_DIAGNOSTIC.json').read_text())
    candidates=[name for name in (best+'_300' if best else None,'FINAL_LONG') if name in summaries]
    usable=[name for name in candidates if summaries[name]['goals_completed_per_episode']>=.75*summaries['CONTROL_300']['goals_completed_per_episode'] and summaries[name]['drop_rate']<=summaries['CONTROL_300']['drop_rate']+.10]
    recommended=max(usable,key=lambda name:summaries[name]['gait/successful_gait_cycles_per_episode']) if usable else 'ROOT'
    result['recommended_demonstration']=recommended
    checkpoint=summaries[recommended]['checkpoint'];mode='none' if recommended=='ROOT' else state['training'][best]['mode']
    gate=False if recommended=='ROOT' else state['training'][best]['gate']
    command=['/home/yoba/Documents/work/IsaacLab/env_isaaclab/bin/python','scripts/rsl_rl/play.py','--task','DG5F-Cube-Stream-Direct-v0','--num_envs','1','--checkpoint',checkpoint,'--real-time','--wait_for_gpu','env.goal_stream_stage=A','env.goal_marker=true',f'env.gait_observation_mode={mode}',f'env.gait_task_gate={str(gate).lower()}','env.gait_transition_fraction=0.0']
    (OUT/'DEMO.sh').write_text('#!/bin/bash\nset -e\ncd '+shlex.quote(str(ROOT))+'\nexec '+shlex.join(command)+'\n')
    (OUT/'final_results.json').write_text(json.dumps(result,indent=2))
    lines=['# Gait decision tree — measured results','',f'Mechanism selected from replicated smoke evidence: **{conclusion}**. True finger gaiting emerged: **{gait}**.', '']
    lines+=['Isolated strict gait cycles already exist in ROOT. NO means no demonstrated emergence of a more organized gait distribution from the tested interventions, not zero individual relocations.','']
    if selected:lines+=['PARTIAL means more measured rare release-transfer-recontact cycles, not a reliable repeated gait. Grasp-mechanics awareness is the strongest tested candidate; a dominant causal bottleneck is not established. The long run has no equally trained control.','']
    if not selected:lines+=['**No long-run candidate justified.** No tested branch met the predefined replicated gait and task criteria. A numerical best short branch is reported for diagnosis, not promoted as a winner.','']
    if state.get('selection_review'):
        lines+=['## Selection review','',state['selection_review']['reason'],'','The original >=20% automatic rule was NOT passed. This long run is exploratory and follows the user criterion after explicit post-smoke review; no measurement thresholds were changed.','']
    lines += ['## ROOT and fixed controls','',f"ROOT: `{state['root_checkpoint']}`.",'All deterministic evaluations use stable resets; training keeps 70/30 stable/transition resets. Physics, control, thresholds and reward weights are unchanged except the explicitly tested task gate. Source tensors, Adam, iteration, LR, std, actor parameter hash and deterministic actor outputs are verified before optimization.','', '## Comparison table','', '| branch / age | seeds | cycles/ep | cycle rate | meaningful/ep | recovery | p90 mm | 2-tip s | goals/ep | drop |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,s in summaries.items():
        if name.endswith('_0') or name.endswith('_150') or name.startswith('AUDIT_'):continue
        vals=[s[k] for k in ('gait/successful_gait_cycles_per_episode','gait/successful_gait_cycle_rate','gait/meaningful_events_per_episode','gait/recovery_to_3plus_rate')]
        lines.append(f"| {name} | {len(s['seeds'])} | "+' | '.join(f'{v:.3f}' for v in vals)+f" | {s['gait/contact_switch_distance_p90']*1000:.2f} | {s['gait/two_tip_support_duration']:.3f} | {s['goals_completed_per_episode']:.3f} | {s['drop_rate']:.3f} |")
    lines+=['','## Hypotheses and selection','']
    for label in ['H2_GATE','H3_CONTACT_MASK','H2_H3','H3_GEOMETRY','H4_GRASP_MECHANICS','H4_H2','H5_CONTACT_MASK_TIMERS']:
        ds=[d for d in state['decisions'] if d['HYPOTHESIS']==label]
        if ds:
            for d in ds:lines+=[f"- **{label}: {d['VERDICT']}**. {d['WHY']} Matched control: {d['MATCHED CONTROL']}. Checkpoint: `{d['CHECKPOINT']}`."]
        else:lines+=[f'- {label}: not tested. '+('No placement-specific failure evidence established by flags; support mechanics prioritized.' if label=='H3_GEOMETRY' else 'Conditional branch was not justified or an earlier candidate qualified.')]
    lines+=['','Selection thresholds were written before training in `EXPERIMENT_DESIGN.json`: goals ≥75% control, drop ≤control+0.10; cycles gain ≥max(0.05,20% control), positive on ≥2/3 smoke seeds and ≥6/8 full-panel seeds, improved cycle rate. H2 additionally requires consistent recovery improvement. H5 is compared to contact flags at equal age. Reward is never the selection score.','', '## Full-panel paired validation','']
    for name,c in comparisons.items():
        if '_full_' not in name:continue
        lines+=[f"{c['branch']} versus {c['matched_control']}: positive={c['positive']}; seeds={c['seeds']}.",'','| metric | paired mean | bootstrap 95% interval | positive seeds |','|---|---:|---|---:|']
        for m,v in c['paired'].items():lines+=[f"| {m} | {v['mean_difference']:.5f} | {v['ci95']} | {v['seeds_positive']}/{v['seeds_total']} |"]
        lines+=['']
    lines+=['## Long-run learning curve','']
    if final_pairs:
        lines+=['Full8-seed final comparisons below are descriptive: training age differs. They do not isolate the effect of H4 from additional training.','', '| reference | metric | paired difference | bootstrap95% interval | positive seeds |','|---|---|---:|---|---:|']
        for reference,metrics in final_pairs.items():
            for k in ['gait/successful_gait_cycles_per_episode','cycles_per_second','goals_completed_per_episode','drop_rate']:
                v=metrics[k];lines+=[f"| {reference} | {k} | {v['mean_difference']:.5f} | {v['ci95']} | {v['seeds_positive']}/{v['seeds_total']} |"]
        lines+=['']
    if not state.get('long_curve'):lines+=['Not run: no replicated qualifying mechanism.','']
    else:
        for x in state['long_curve']:lines += [f"- +{x['additional']}: `{x['checkpoint']}`, task/gait guardrails={x['useful']}."]
    if result.get('H4_time_normalized_diagnostic'):
        d=result['H4_time_normalized_diagnostic']['paired_H4_minus_control']['cycles_per_second']
        lines+=['','H4 smoke gains partly reflect longer survival: cycles per second improve by about8.5%, positive in6/8 seeds, but their paired bootstrap interval includes0. Exact time-normalized results: `H4_TIME_NORMALIZED_DIAGNOSTIC.json`. This limits the mechanistic conclusion.','']
    lines+=['','## Final control/contact metrics','', '| branch | 1 tip | 2 tips | ≥3 tips | support-qualified goals | raw μ | saturation | std |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in dict.fromkeys(['ROOT','CONTROL_300',*(['H4_GRASP_MECHANICS_300'] if 'H4_GRASP_MECHANICS_300' in groups else []),bestname]):
        s=summaries[name];lines += [f"| {name} | {s['tip_fraction_1']:.3f} | {s['tip_fraction_2']:.3f} | {s['tip_fraction_3plus']:.3f} | {s['support_qualified_goals_per_episode']:.3f} | {s['raw_mu']:.3f} | {s['saturation']:.3f} | {s['action_std']:.3f} |"]
    for name,d in diagnostics.items():
        lines+=['',f'### {name}: contact graph and failure phases','',f"Most common transitions: `{d['top_transitions'][:8]}`.",f"Most common return loops: `{d['top_return_loops'][:8]}`.",f"Event filters (overlapping, not a causal waterfall): `{d['event_filters']}`.",f"Mean normalized leave-one-out quality before release, by finger (zero indexed): `{d['mean_leave_one_out_at_release_by_finger']}`.",'','| finger | releases | closed recontacts | meaningful | strict cycles | support fraction | mean displacement mm |','|---|---:|---:|---:|---:|---:|---:|']
        for f,v in d['per_finger'].items():lines += [f"| {int(f)+1} | {v['release_count']} | {v['events']} | {v['meaningful']} | {v['successful_cycles']} | {v['support_frequency_sum']/len(d['seeds']):.3f} | {v['displacement_mean_sum']/len(d['seeds'])*1000:.3f} |"]
        if d['qdot_per_seed']:lines += ['',f"Finite-difference qdot p99 across seeds: {[round(v['qdot_fd_after_1s'].get('p99',0),3) for v in d['qdot_per_seed']]} rad/s. PhysX and finite-difference distributions are both preserved in final_results.json."]
    cache=json.loads((ROOT/'logs/gait_research/cache_composition.json').read_text())
    lines+=['','## Goal-stream diagnostic','', 'Stage A samples a random sign for each20-degree target (`goals.py:161`). Thus commanded_rotation_deg is cumulative requested travel, not net rotation or completed one-direction turns. The current fixed task can reward reversals. This suggests a task-demand explanation for weak gait pressure, but does not establish it; no target-generation change was made tonight.', 'For LONG+500, the saved384 trajectories have mean234.4deg commanded travel, mean50.0deg absolute integrated signed body-x motion, and1443.8deg unsigned angular path after the first second. The body-x integral is not a global-axis net orientation, and angular path includes jitter. Full definitions and per-episode data: `REVERSIBLE_GOAL_DIAGNOSTIC.json`.','']
    lines+=['','## Transition-cache coverage','',f"Cache unchanged: {cache['robust_snapshots']} snapshots from {cache['robust_transitions']} physical transitions / {cache['successful_source_grasps']} source grasps among {cache['attempted_source_grasps']} attempted. Snapshot counts by moving finger (zero indexed): {cache['by_finger']}. Phase counts: {cache['states_by_phase']}. This is narrow coverage, particularly for fingers absent from the successful scripted transitions; it limits conclusions about the observation hypotheses."]
    audit_path=OUT/'DELIVERED_ACTION_AUDIT.json'
    if audit_path.exists():
        lines+=['','## Delivered-action audit','', 'Additional matched seed31415,128 episodes each. Issued commands and actual post-FIFO commands are measured separately. Near-limit means absolute action>=0.95 in the clipped[-1,1] range; this is distinct from motor torque saturation.','', '| policy | issued saturation | delivered saturation |','|---|---:|---:|']
        for name,v in json.loads(audit_path.read_text()).items():lines+=[f"| {name} | {v['issued_near_limit']:.5f} | {v['delivered_near_limit']:.5f} |"]
    lines+=['','## Demonstration command','',f"Recommended measured checkpoint: `{checkpoint}` ({recommended}). Run `bash logs/gait_tree/DEMO.sh` from the project root. The script waits for the GPU lease and passes the observation mode required by this checkpoint.",'']
    lines+=['','## Videos','']+[f'- [{v}]({v})' for v in result['videos']]
    lines+=['','Same camera, seed31415, deterministic actions, stable resets, up to3 episodes /60s. Videos are sanity evidence; numerical verdict uses128 episodes per evaluation seed.','', '## Limits and next experiment','']+['- '+v for v in result['limitations']]
    if (OUT/'VISUAL_REVIEW.md').exists():lines+=['','Visual review: [VISUAL_REVIEW.md](VISUAL_REVIEW.md). Sampled frame sequences verify rendering and changing grasp configurations; occlusion prevents certifying every strict gait condition from video alone.','']
    lines+=['','Next experiment: replicate the selected mechanism with a second independent training seed and a matched long control.' if selected else 'Next experiment: a controlled imitation warm-up using the validated scripted release-transfer-recontact trajectories, then PPO with unchanged rewards; compare against equal PPO updates. This tests action discovery directly after the tested semantic/observation changes failed.','',f"Git HEAD: `{result['initial_head']}` → `{result['final_head']}`. No commits performed by this campaign."]
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        names=list(dict.fromkeys(['ROOT','CONTROL_300',bestname]));fig,axes=plt.subplots(2,2,figsize=(10,7))
        plotkeys=[('gait/successful_gait_cycles_per_episode','Successful gait cycles / episode'),('gait/meaningful_events_per_episode','Meaningful relocations / episode'),('goals_completed_per_episode','Goals / episode'),('drop_rate','Drop rate')]
        for ax,(key,title) in zip(axes.flat,plotkeys):
            for seed in sorted(set.intersection(*(set(r['seed'] for _,r in groups[n]) for n in names))):
                vals=[next(r[key] for _,r in groups[n] if r['seed']==seed) for n in names]
                ax.plot(range(len(names)),vals,'o-',alpha=.4,linewidth=1)
            ax.set_xticks(range(len(names)),names,rotation=12);ax.set_title(title);ax.grid(alpha=.2)
        fig.suptitle('Stable-reset deterministic evaluation: lines pair evaluation seeds');fig.tight_layout();fig.savefig(OUT/'paired_comparison.png',dpi=150);plt.close(fig)
        lines+=['','![Paired evaluation comparison](paired_comparison.png)']
        if state.get('long_curve'):
            start=[r for _,r in groups[selected+'_300'] if r['seed'] in (1234,4321,2026)]
            curve=[{'additional':0,'rows':start}]+[{**v,'rows':[r for _,r in groups['LONG_'+str(v['additional'])]]} for v in state['long_curve']]
            fig,axes=plt.subplots(2,3,figsize=(14,7))
            for ax,(key,title) in zip(axes.flat,[plotkeys[0],('cycles_per_second','Gait cycles / second'),('episode_length_s','Episode duration, seconds'),plotkeys[2],plotkeys[3],('action_std','Exploration standard deviation')]):
                x=[v['additional'] for v in curve];y=[statistics.mean(r[key] for r in v['rows']) for v in curve]
                sd=[statistics.stdev(r[key] for r in v['rows']) for v in curve]
                ax.errorbar(x,y,yerr=sd,fmt='o-',capsize=3);ax.set_title(title);ax.set_xlabel('Additional H4 updates');ax.grid(alpha=.2)
            fig.suptitle('Long H4: mean and evaluation-seed SD (3 fixed seeds)');fig.tight_layout();fig.savefig(OUT/'long_learning_curve.png',dpi=150);plt.close(fig)
            lines+=['','![Long H4 learning curve](long_learning_curve.png)']
    except ImportError:
        lines+=['','Plot unavailable; complete numerical tables are preserved.']
    (OUT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
