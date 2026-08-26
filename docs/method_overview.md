# Method overview

OPAL treats the federated clean-label backdoor problem as a signal-transfer problem.

The experiment pipeline is:

1. build clean-label poison samples without label flipping;
2. train local benign and malicious clients;
3. construct bounded submitted updates;
4. trace whether the trigger-loss descent direction survives submitted-update scaling;
5. aggregate updates with FedAvg;
6. continue benign-only recovery after attack stop;
7. evaluate clean accuracy, ASR@stop, and ASR-t.

The most useful mechanism from the experiments is submitted-update write-in:

```text
u_new = ||u|| * normalize((1 - alpha) * normalize(u) + alpha * normalize(-g_trigger))
```

Here `u` is the actual submitted client update and `g_trigger` is the source-trigger-to-target loss gradient. This directly edits the aggregation-facing update while preserving its norm.

Stage5r extends this idea by changing where the write-in acts, not how strong it is: the first gate restricts write-in to `layer4.` and `linear.` parameters.

