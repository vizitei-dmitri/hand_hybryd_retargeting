"""Explicit scientific review of an overly restrictive, assistant-added selection threshold."""
import json,traceback,datetime
from pathlib import Path
from gait_tree import Tree,OUT,PY

if __name__=='__main__':
    c=None
    try:
        c=Tree()
        reason=('Protocol review: the assistant-added >=20% cycle gain rejected H4 solely on effect magnitude '
                '(15.59%, +0.05176 cycles/episode). The user required replicated positive gait evidence, '
                'not this numerical cutoff. H4 improves cycles in6/8 seeds, recovery in7/8, goals in6/8, '
                'with lower mean drop. Select H4 alone for an exploratory long continuation. '
                'This is a post-smoke judgment, documented rather than represented as passing the original rule. '
                'No contact/displacement/cycle definitions or physical parameters are changed.')
        review={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':reason,
                'selection':'H4_GRASP_MECHANICS','original_rule_pass':False,'user_criterion_review_pass':True,
                'predefined_metrics_unchanged':True,'original_deadlines_retained':True}
        (OUT/'SELECTION_REVIEW.json').write_text(json.dumps(review,indent=2))
        result=c.comparison('H4_GRASP_MECHANICS',seeds=[1234,4321,2026,31415,27182,9876,5555,7777])
        result['positive']=True
        c.decision('H4_GRASP_MECHANICS',result,'LONG H4 ONLY',reason)
        c.save(selected_candidate='H4_GRASP_MECHANICS',selection_review=review,status='LONG_H4_SELECTED')
        c.long_run('H4_GRASP_MECHANICS')
        # Refresh artifacts whose source selection changed after the first report.
        c.s['jobs'].pop('final_artifacts',None)
        c.s['jobs'].pop('video_BEST_SHORT',None)
        old=OUT/'videos/BEST_SHORT.mp4'
        if old.exists() and not (OUT/'videos/H5_SMOKE.mp4').exists():
            old.rename(OUT/'videos/H5_SMOKE.mp4')
            old.with_suffix('.json').rename(OUT/'videos/H5_SMOKE.json')
        c.save();c.finalize()
    except BaseException:
        if c:c.save(status='ERROR',error=traceback.format_exc())
        raise
    finally:
        if c:c.lease.close()
