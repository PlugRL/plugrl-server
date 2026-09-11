# E3 measurement protocol (pre-registered)

**Status: written, never executed.** The comparison this protocol describes
was planned for a paper that is no longer being written. It is kept because
the protocol itself is the useful part - it fixes the rules before any data
exists, and it declares the bias that would have favoured us. Anyone
repeating this kind of comparison can start from it. **No numbers were ever
collected, so nothing here should be cited as a result.**

**Written 2026-09-10, before any data was collected.**

---

## Why pre-register

E3 was to claim that adding a new environment or policy costs less on PlugRL
than on a monolithic framework. The weakest point of such a claim is not the
numbers, it is the **definitions**: what counts as "integrated", how lines
are counted, when the clock starts, whether failed attempts count.

If those rules are chosen after seeing the results, a reviewer is right to
suspect they were chosen to fit. So they are fixed first.

**This file must not be edited after the first data point is collected.** If
a rule turns out to be flawed, write a new version explaining why the old one
was unusable, and keep both.

---

## Task definitions

Two task types, measured independently.

### T-ENV: integrate a new environment

**Done when** the environment completes 3 episodes with an existing policy,
under 3 different seeds, **all of them**. Not finishing is not finishing.

### T-POL: integrate a new policy

**Done when** the policy completes 3 episodes x 3 seeds on an existing
environment and produces a non-zero gradient update - that is, `learn`
executes successfully at least once.

**Note:** it is not required to learn anything. E3 measures integration cost,
not learning quality; correctness is E4's job.

---

## The three subjects

| Framework | Version pinning |
|---|---|
| PlugRL | a specific commit of this repository (record the SHA) |
| RLinf | the release tag on the day of collection |
| SimpleVLA-RL | the release tag on the day of collection |

The comparison versions must be taken on the **same day**, with SHAs or tags
recorded, and must not be updated partway through.

---

## Four metrics

### 1. Lines added or modified

* **Count only lines a human writes**: all lines of new files, plus the sum
  of additions and deletions reported by `git diff --numstat` for existing
  files.
* **Do not count** generated files (lock files, `__pycache__`), comments or
  blank lines - use the `code` column of `cloc --by-file`, not `wc -l`.
* **Do not count** formatter-induced changes: run each project's own
  formatter before measuring.
* **Record separately**: changes to the framework itself versus code added on
  the user side. Touching the framework is the more expensive kind, because
  it means a fork or an upstream pull request.

### 2. New dependencies

* Taken from each framework's lock file: the difference in the resolved
  package set from `uv.lock` / `poetry.lock` / `requirements.txt` before and
  after.
* **Count additions only**, not upgrades.
* Note separately whether a **new system-level dependency** was introduced
  (an apt package, CMake, a compiler, a CUDA version requirement). That kind
  of cost is far higher than a wheel.

### 3. Wall clock

* **Starts** the first time an editor or terminal is opened for the task.
* **Ends** the moment the completion criterion is satisfied.
* **Includes** reading documentation, trial and error, installing
  dependencies, debugging.
* **Excludes** pure download waiting (record it, report it separately) and
  interruptions unrelated to the task.
* Keep a simple log: one line per start and pause in `timing.jsonl`, with a
  timestamp and a sentence.

**Known bias:** the operator is familiar with PlugRL and unfamiliar with
RLinf and SimpleVLA-RL. This systematically makes PlugRL look cheaper. See
below.

### 4. Whether the framework itself had to change

A boolean plus an explanation. Reported separately, not folded into the line
count.

---

## Bias and mitigation

### Familiarity bias (the serious one)

**The problem**: we wrote PlugRL and have never written RLinf. One person
doing all three sides guarantees the time metric favours PlugRL.

**Mitigations, strongest first:**

1. **Preferred**: have someone who has **never worked on PlugRL** do all
   three, keeping their own time. This is the only option that actually
   removes the bias.
2. **Second**: one person does all three, but does **RLinf and SimpleVLA-RL
   first and PlugRL last**, so the learning effect works against PlugRL.
3. **Fallback**: if only option 3 is available, **demote the time metric to
   supporting evidence**. The main conclusion then rests on lines and
   dependency counts, which are far more objective, and the limitation is
   stated explicitly.

**Whichever is used must be named in the write-up.**

### Selection bias

The environments and policies integrated must not be chosen to suit PlugRL.
**The sample is fixed here, before collection:**

* T-ENV: `RoboCasa`, `ManiSkill` - PlugRL has neither.
* T-POL: `Diffusion Policy`, and a plain-MLP SAC policy.

If a sample **cannot** be integrated into some framework at all, record it as
not completed and say where it stopped. Do not substitute a different sample.

### Stopping rule

Each (framework x task) cell has an **8 hour** limit. Beyond that it is
recorded as "not completed within budget", along with where it stopped. This
stops "one more day might do it" from becoming unbounded.

---

## Record format

Each (framework x task) produces one `<framework>-<task>.json`:

```json
{
  "framework": "plugrl",
  "framework_ref": "4aaebe0",
  "task": "T-ENV",
  "target": "ManiSkill",
  "completed": true,
  "loc_user": 210,
  "loc_framework": 0,
  "new_deps": 4,
  "new_system_deps": ["libvulkan1"],
  "wall_clock_minutes": 95,
  "download_minutes_excluded": 22,
  "required_framework_changes": false,
  "notes": "...",
  "operator": "...",
  "order_index": 3
}
```

`order_index` records the position of the task in the execution order, so the
learning effect can be checked afterwards.

---

## Reporting

* Three frameworks side by side, **no normalisation**, no ratios.
* Each cell marked completed or not, and within budget or not.
* If the fallback mitigation was used, the time metric is explicitly labelled
  as supporting evidence in the table itself.
* **Cells that were not completed are shown anyway**, with the blocker. That
  is a result too.
