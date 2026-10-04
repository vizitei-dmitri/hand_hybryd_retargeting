"""Final matched-seed measurement of actions after the actual delay queue."""
import json,traceback
from pathlib import Path
from gait_tree import Tree,OUT,SOURCE,PY

if __name__=='__main__':
    c=None
    try:
        state=json.loads((OUT/"night_state.json").read_text())
        if state.get("status")!="AWAITING_VIDEO_REVIEW":
            raise RuntimeError("Long controller did not finish cleanly; do not mask its failure")
        c=Tree();targets=[('ROOT',SOURCE,False,'none'),('CONTROL',Path(c.s['training']['CONTROL_LONGISH']['checkpoint']),False,'none')]
        i=c.s['training']['H4_GRASP_MECHANICS'];targets.append(('H4_SHORT',Path(i['checkpoint']),False,'mechanics'))
        if c.s.get('best_long_checkpoint'):targets.append(('BEST_LONG',Path(c.s['best_long_checkpoint']),False,'mechanics'))
        result={}
        for label,source,gate,mode in targets:
            name='AUDIT_'+label+'_s31415';c.evaluate(name,source,31415,gate,mode)
            r=next(iter(json.loads((OUT/(name+'.json')).read_text()).values()))
            result[label]={'checkpoint':str(source),'seed':31415,'episodes':r['episodes'],
                           'issued_near_limit':r['smoothing']['command']['near_limit_fraction'],
                           'delivered_near_limit':r['smoothing']['delivered_command']['near_limit_fraction'],
                           'raw_mu_mean_abs':r['smoothing']['raw_mu']['mean'],
                           'qdot_fd_after_1s':r['smoothing']['qdot_fd_after_1s']}
        (OUT/'DELIVERED_ACTION_AUDIT.json').write_text(json.dumps(result,indent=2))
        c.run('final_artifacts_after_delivery_audit',[PY,'scripts/gait_tree_report.py'])
        c.save(status='AWAITING_VIDEO_REVIEW',next_command='Inspect final H4 videos/plots, verify final artifacts and HEAD, then mark COMPLETE.')
    except BaseException:
        if c:c.save(status='ERROR',error=traceback.format_exc())
        raise
    finally:
        if c:c.lease.close()
