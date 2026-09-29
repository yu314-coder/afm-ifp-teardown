# Apple Developer Forums — draft post (Core ML / ML Compute)

**Title:** Is there a supported way to capture per-node intermediate outputs from an ANE-scheduled model?

**Tags:** Core ML, ML Compute, Machine Learning

---

I'm looking for a supported way to read intermediate tensors from a model executing on the Apple
Neural Engine — specifically the output of an individual node in the compiled graph, rather than
just the final output.

What I'm trying to do: validate a from-weights reimplementation of a model against the real thing,
layer by layer. Final-output comparison tells me the reimplementation is wrong but not *where*,
and a per-layer comparison would localise it immediately.

What I've established so far:

- A compiled ANE program can be executed unprivileged through the public graph API, and the
  final readout matches, so the execution path itself is reachable.
- Intermediate activations do not appear in host memory during normal operation — which is
  expected, since the scheduler keeps them in accelerator-local storage.
- Requesting a per-node output appears to hit a kernel-side check that isn't satisfied by an
  ordinary process.

Questions:

1. Is there a supported API for retrieving per-node outputs from an ANE-scheduled graph — a debug
   or instrumentation mode, an Instruments template, or a Core ML compute-plan facility that
   surfaces them?
2. Failing that, is there a supported way to force a specific node to materialise its output to a
   host-visible buffer — for example by splitting the graph, marking a tensor as an output, or
   compiling with the node as a terminal operation?
3. If neither exists, is that a deliberate design boundary rather than a gap? A clear "no" is a
   useful answer and I'll stop looking.

I'm not asking about any particular shipped model, and this isn't a request to bypass anything —
I'm asking whether the platform exposes per-node observability for ANE execution at all, and if
so, what the supported entry point is.

Thanks.

---

## Notes (do not post)

- **Registration first.** The forums require a one-time profile with a public username. Choose
  something you're happy having attached to this permanently.
- **Deliberately generic.** No mention of Apple Intelligence, the FM assets, or the teardown. The
  question stands on its own as a platform-observability question, which is what makes it
  answerable. Naming the model invites the thread to be closed as out of scope, and the answer you
  want — does per-node capture exist — is the same either way.
- **Do NOT add the weight-publication question to this post.** See below.

## Why the weights question does not belong on the forums

Three reasons, in order of weight:

1. **Nobody there can answer it.** The forums are staffed by Apple engineers answering API
   questions. Licensing and IP are not theirs, and they are explicitly not permitted to give legal
   guidance. The question would be ignored, or the thread removed.
2. **It would damage the position you have built.** Publicly asking "may I publish Apple's model
   weights" reframes three careful disclosures — in which you repeatedly stated you had not
   published and would not publish any Apple weights — as groundwork for redistribution. That is
   the single sentence from this project most likely to be quoted out of context.
3. **The paper does not need it.** Citing recovered weights does not require republishing them.
   You cite the shipped asset by name, build, and hash, and you publish the decoder — which is
   already public — so any reader with the asset on their own device can regenerate the tensors
   and check every number. That is the reproducibility standard for a format teardown, and this
   work already meets it.

## What you actually have on the publication question

Three written determinations from Apple Product Security, obtained across two channels:

- OE11069002425417 (28 July, Nick): reconstructing the assets "does not expose anything a device
  owner could not already access."
- Email thread OE01069008668316 (11 August, Robert): "The confidentiality of on-device model
  weights is not among the security/privacy boundaries our program covers."
- OE1107308765845 (27 August): closed, no security issue identified.

None is permission to publish Apple's copyrighted material, and none should ever be cited as
though it were. What they do establish, and what the paper can properly say, is that Apple was
notified three times, was shown the public repository, and raised no objection to the analysis
being published. For a teardown that publishes no weights, that is a stronger disclosure record
than most published reverse-engineering carries.
