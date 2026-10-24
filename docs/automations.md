# Automations

An automation runs a project's chapters without you: by hand, once at a set time, or on
chosen days at a set time. Open one from the calendar button next to the project name.

## What a run does

1. **Downloads, if asked.** With *Download chapters from the linked source first*, the run
   refreshes the linked series and downloads the chapters the project lacks, in reading
   order (one copy per chapter number), at most the chapter limit.
2. **Picks the chapters.** Either the chapters you chose, or every chapter with a page that a
   chosen stage has not processed yet, in project order, at most the limit. A chapter a run
   finished without errors is not picked again (*Forget finished chapters* undoes that).
3. **Runs one stage at a time.** Detection on every page, then OCR on every page, then
   translation, then inpainting. Every model is unloaded and the GPU cache released before
   the next stage loads its model, so each model has the GPU to itself.
4. **Keeps going past errors.** A page that fails a stage is recorded and left out of the
   later stages that need that stage. A page whose translation failed is still inpainted.
5. **Reports.** The run history lists each chapter as done, partly done or failed, with the
   errors and the model each stage ran.

A run is ordinary pipeline work: each page's result is its own undo step, user edits are
never overwritten, and running again continues where the last run stopped.

## Model profiles

A profile names the models a run uses instead of your current settings, per stage; an empty
stage keeps the setting. Because stages run one at a time, a profile can name models that
would not fit on the GPU together, such as a large translation model and a diffusion
inpainting model. Profiles are shared by every project.

## Scheduling

- Times are local. *Repeat* runs on the chosen weekdays; *Once* runs at one date and time,
  then turns itself off.
- Urakatsu must be running (the window can be minimized). A run that came due while it was
  closed starts when it opens, if it is at most six hours late; otherwise it waits for the
  next time.
- One run goes at a time, since there is one GPU. Others queue. A run waits for a run you
  started by hand to finish; while a run is going, starting one by hand is refused.
- The computer is kept from sleeping while a run is going.
- **Wake the computer** (Windows) registers one Task Scheduler entry,
  `Urakatsu\Automations-<id>`, set to the next automation that asked for it. It wakes the
  computer shortly before that time and starts Urakatsu minimized if it is closed. The entry
  is removed when no automation needs it.

## Storage and API

Automations and their last 30 runs live in `projects/<id>/automations.json`; profiles in
`URAKATSU_HOME/automation-profiles.json`.

| Route | Purpose |
| --- | --- |
| `GET /api/v1/automations/status` | The running run, the queue, and whether waking is supported |
| `GET`, `POST /api/v1/automations/profiles`; `PUT`, `DELETE .../{profile}` | Model profiles |
| `GET`, `POST /api/v1/projects/{id}/automations`; `PUT`, `DELETE .../{automation}` | A project's automations and runs |
| `POST /api/v1/projects/{id}/automations/{automation}/run` | Queue a run now |
| `POST /api/v1/projects/{id}/automations/{automation}/reset` | Forget the chapters it finished |
| `POST /api/v1/automations/runs/{run}/stop` | Stop a running run, or drop a queued one |

The code is `urakatsu/pipeline/batch.py` (the stage-at-a-time runner),
`urakatsu/service/automations.py` (schedules, queue, runs) and `urakatsu/core/power.py`
(keeping the computer awake, the wake task). Koharu has no equivalent.
