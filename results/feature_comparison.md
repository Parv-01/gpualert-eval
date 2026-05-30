# Feature comparison

knockknock isn't a baseline in Experiment 1, and on purpose: it's a notification
tool, not a failure classifier, so there's nothing to put in a confusion matrix
for it. The fair way to compare against it (and the other things people reach for)
is at the feature level. This is the table that belongs in the paper's Related
Work, next to the prose differentiation.

| capability | gpualert | knockknock | Slurm `--mail-type` | NVIDIA DCGM |
|---|---|---|---|---|
| Notifies on completion / crash | yes | yes | yes (job state) | no (health telemetry) |
| Zero-instrumentation (wrap any command, no code edits) | yes (process wrapper) | no (decorate your code) | yes (scheduler config) | yes (daemon) |
| Classifies the failure cause | yes (15 modes) | no | no | partial (GPU/XID hardware only) |
| Remediation suggestions | yes | no | no | partial (hardware) |
| Durable pre-launch log + delivery guarantee | yes | no | no | n/a |
| Attaches logs / artifacts to the notification | yes | no (message only) | no | no |
| Slurm `sacct` integration | yes | no | native | no |
| Notification channels | email (extensible) | 12+ platforms | email only | n/a |
| What it watches | app + job failures | app crash / completion | job state transitions | GPU hardware health |
| Runtime footprint | per-job wrapper | import + decorator | scheduler-side | root daemon |

Reading it honestly: knockknock wins on channel breadth (it speaks Slack,
Discord, Telegram and a dozen others out of the box, where gpualert is
email-first). DCGM isn't really a competitor at all -- it watches the hardware,
gpualert watches the job, and the two are complementary; a serious cluster runs
both. Slurm's `--mail-type` is the exit-code baseline from Experiment 1 wearing a
different hat: it tells you a job failed, nothing about why.

Where gpualert is alone is the combination in the middle rows: zero-instrumentation
*and* a 15-way cause classification with remediation *and* a log-durability
guarantee that survives a crash, for any wrapped command. None of the others
occupy that point. That's the positioning sentence for the paper:

> GPUAlert occupies a distinct point: unlike heavyweight cluster managers that
> require platform integration, and unlike decorator-based notifiers that require
> code modification, it provides zero-instrumentation, classification-augmented
> failure notification for any wrapped command, with a log-durability guarantee.

The quantitative backing for the "classification" claim is Experiment 1 (gpualert
macro-F1 0.905 vs. the exit-code baseline, i.e. `--mail-type`, at 0.133); for the
"durability" claim it's Experiment 2.
