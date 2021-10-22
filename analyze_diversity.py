import pickle
import collections
import numpy as np
from eval_metrics import evaluate_metrics


def eval_distinct_k(candidates, k):
  """The total number of k-grams divided by the total number of tokens
     over all the candidates.
  """
  kgrams = set()
  total = 0
  for cand in candidates:
    if len(cand) < k:
      continue

    for i in range(0, len(cand)-k+1):
      kgrams.add(tuple(cand[i:i+k]))
    total += len(cand)
  if total == 0:
    print('Why does this happen sometimes?')
    import pdb; pdb.set_trace()
  return len(kgrams) / total

def eval_entropy(candidates, k):
  """Entropy method which takes into account word frequency."""
  kgram_counter = collections.Counter()
  for cand in candidates:
    for i in range(0, len(cand)-k+1):
      kgram_counter.update([tuple(cand[i:i+k])])

  counts = kgram_counter.values()
  s = sum(counts)
  if s == 0:
    # all of the candidates are shorter than k
    return np.nan
  return (-1.0 / s) * sum(f * np.log(f / s) for f in counts)


algo_path = 'outputs/eval_metrics/cluster_beam_2_2/'

with open(algo_path + 'captions_gt.pkl', 'rb') as f:
    captions_gt = pickle.load(f)




with open(algo_path + 'captions_pred.pkl', 'rb') as f:
    captions_pred = pickle.load(f)

candidates = [c['caption_predicted'].split() for c in captions_pred]


metrics = evaluate_metrics(captions_pred, captions_gt)


for metric, values in metrics.items():
    print(f'{metric:<7s}: {values["score"]:7.4f}')

print('dist-1 : {}'.format(eval_distinct_k(candidates, 1)))
print('dist-2 : {}'.format(eval_distinct_k(candidates, 2)))
print('ent-1 : {}'.format(eval_entropy(candidates, 1)))
print('ent-2 : {}'.format(eval_entropy(candidates, 2)))


