# Disclosure record — Apple Product Security

A dated record of the disclosure trail for this work, so the published research can be checked
against what Apple was told and when. Apple's replies are quoted only in the operative sentences
that state their determination; the full messages are retained privately and are not reproduced
here.

**Portal cases:** `OE11069002425417` (closed), `OE1107308765845` (reopened by a comment, Received), `OE1104472956171` (permission request, Received)
**Original title:** `OE11069002425417` — *Weight recoverability in the shipped Apple Intelligence
on-device model assets*
**Email thread reference:** `OE01069008668316`

---

## 1. Initial disclosure — 26 July 2026

Submitted twice, in both channels: by email to `product-security@apple.com`, and through
`security.apple.com/submit` (an automated reply to the email directed that security issues be
submitted online, so the portal submission is the tracked one).

Substance: the shipped `com.apple.MobileAsset.UAF.FM.GenerativeModels` assets are sufficient to
reconstruct the on-device models' weight tensors — the storage codec is fully reversible, the
parameter budget closes to 100.00% of the shipped bytes, and the decoder validates by round-trip
against `ANECompiler` at correlation 0.981. The report also flagged that two apparent barriers
(the IFP router→expert map and a "missing constant table") do not in fact protect the models, and
raised as time-sensitive that a `..._3B_EMBEDDINGS_..._Cryptex.dmg` had begun shipping in build
26A5378n where the embedding table had previously been absent.

The report stated that no signing material was bypassed and no protection was circumvented, that
no Apple weights, tokenizer data, or assets had been or would be published, and offered to delay
further publication pending Apple's assessment. The public repository was linked in the report.

The letter as sent is preserved at [`LETTER_TO_APPLE.md`](LETTER_TO_APPLE.md).

## 2. Apple's determination — 28 July 2026

**Nick | Product Security**, closing the portal case:

> "After review we determined this does not represent a security vulnerability, because the model
> assets involved are already readable on the device and reconstructing them does not expose
> anything a device owner could not already access."

## 3. Apple's determination — 11 August 2026

**Robert | Apple Product Security**, on the email thread:

> "we do not see any reasons to track this as a security issue. As you note in your own report, no
> signing material was bypassed and no protection was circumvented; the analysis reads shipped,
> already-unencrypted assets that are accessible to the device owner. The confidentiality of
> on-device model weights is not among the security/privacy boundaries our program covers."

and:

> "We appreciate the rigor and honesty of your write-up and hope you'll continue to share findings
> with us in the future."

## 4. Correction filed — 26 August 2026

Posted as a comment on case `OE11069002425417`, taking up the invitation above and the portal's
own note that new information may allow a closed report to be reviewed further.

Substance: a **downgrade of the original report**. Weight recoverability still holds, but
recovering the weights turned out **not to be sufficient to reproduce the model**. A working
transformer converts depth into ≈ −7.7 nats of predictive value on held-out text; the
reconstruction converts depth into **+0.47**, i.e. its layers destroy information the embedding
alone already carries. 42 controlled experiments eliminated every hypothesis constructible from
the shipped artifacts. The evidence points to a basis disagreement on the shared residual axis,
which is not resolvable from outside because every available static statistic is invariant to a
permutation of that axis.

The comment states explicitly that this **supports** Nick's closing rationale rather than
challenging it: the practical barrier sits further out than the original report implied, because
the accelerator's layout conventions are not recoverable from the shipped bytes and turn out to be
load-bearing.

It also asks, flagged as *not* a security issue and only because this is the channel that has
replied, whether a supported path exists for per-node activation capture on owned hardware, or
which team fields research-access questions of that kind.

Full technical treatment: [`../paper/reconstruction.tex`](../paper/reconstruction.tex) —
*The Reconstruction Gap: A Controlled Negative Result*.

**Filing this comment reopened the case** (status `Closed` → `Received`).

## 5. Publication question — 26 August 2026

Posted as a second comment on the same case. Asks directly whether Apple objects to the findings
remaining published on GitHub, stating plainly that the repository has been public since the July
report and was linked in it, rather than asking as though nothing had happened.

Itemised as published: container and asset layouts, the 2-bit codec and its per-row scale rule,
the ANE tile permutation, the `odix` IR grammar, recovered architecture, the author's own decoder
and analysis code, measurements derived from the weights, and the reconstruction failure with its
controls.

Itemised as never published: Apple weights in any form including decoded tensors and any GGUF or
`state_dict` built from them; the tokenizer vocabulary; any shipped asset, cryptex, `hwx`, `odix`,
or `mpsgraph` file. The repository enforces this with a commit filter on those file types.

The comment acknowledges this may not be Product Security's call, asks to be routed rather than
treating silence as agreement, and offers to remove specific material if asked.

## 6. Re-filed as an open report — 26 August 2026

**New case:** `OE1107308765845` — *Follow-up to OE11069002425417 - severity correction and a
publication question* (status: Received)

The two comments in §4 and §5 were added to case `OE11069002425417`, which stayed **Closed**; the
portal's own note says new information "may allow us to review your report further", but the case
was not reopened. The same content was therefore re-filed as a new report so that it reaches a
live queue.

The submission states in its first line that it is **not a claim of a vulnerability**, records that
it duplicates comments already made on the closed case, and asks Apple to close it if comments on
closed reports are in fact routed. Affected area is set to *Not Listed*; the proof-of-concept
field says there is no exploit and no issue to demonstrate, and that the report is **not eligible
for and not submitted for the Apple Security Bounty**.

Contents: (1) the severity correction from §4, (2) the publication question from §5, (3) the
activation-capture question, flagged as not a security issue.

---

## Follow-up case closed — status as of 8 November 2026

The portal shows case `OE1107308765845` as **Closed**, with the portal's standard status text:
"We're unable to identify a security issue in your report." Like the original case, it invites new
information as a comment.

On 8 November 2026 (18:04) a comment was added to that case. It asks Product Security to forward the
licensing question (sent to Apple Legal the same day through the Rights and Permissions form) to the
right contact, or to name that contact. It restates that only original code and measurements are
published, and that nothing derived from the weights will be published until Apple replies.

---

## Licensing request to Apple Rights and Permissions — 8 November 2026

Separate from the security cases, a permission request was submitted through apple.com
(Legal → Contact → *Rights and Permissions*, the channel for licensing Apple copyrighted material).
It states that the converter and llama.cpp patch are public in this repository, that no model data
has been published, and asks: (1) whether Apple objects to that code remaining public under the MIT
license, (2) whether the converted weights may be redistributed and on what terms, (3) what Apple
would want changed in a code-only release, (4) whether Apple would publish the weights itself. It
cites the two Product Security cases above. Apple's form confirmed receipt with a standard page
stating that requests receiving no reply from an Apple employee cannot be processed.

---

## Standing position

Apple has twice stated in writing that this work does not raise a security issue for them, and has
not objected to publication. Neither statement is a grant of permission to publish Apple's
copyrighted material, and none has been sought or assumed: **no Apple weights, tokenizer data, or
shipped assets appear in this repository, and none will.** What is published is original analysis,
original code, and measurements — the same category as any published format teardown.

Any reply from Apple will be added here.

---

## New portal case: permission request — 8 November 2026

Because comments on a closed case may not be read, the licensing question was filed as a new report in the
security portal: `OE1104472956171`, *Permission request: open-source release of an AFM reconstruction (not a
security report)*. It states up front that it is not a security report, and asks Product Security to route it or
name the right contact. It asks: (1) whether Apple objects to the code-only repository staying public under the MIT
license, (2) whether the converted model files may be published on GitHub or Hugging Face and on what terms, and
(3) what Apple would want changed in the code-only release. It repeats that nothing derived from Apple's weights will
be published unless Apple agrees. The form's required fields that do not apply ("what is required to reproduce", and
the proof of concept) were answered "Not applicable – this is a permission request". Status: Received. The portal says
reviews usually take about a week. The comment added earlier the same day returned case `OE1107308765845` to Received.
