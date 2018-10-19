# On Github Issues and Pull Requests

Found a bug? Have a new feature to suggest? Want to contribute changes to the codebase? Make sure to read this first.

You can read this [document to learn how to do your first contributions](https://github.com/firstcontributions/first-contributions#first-contributions) (🇫🇷 aussi [disponible en français](https://github.com/firstcontributions/first-contributions/blob/master/translations/README.fr.md)).

## Bug reporting

Your code doesn't work, and you have determined that the issue lies with SMPyBandits? Follow these steps to report a bug.

1. Your bug may already be fixed. Make sure to update to the current SMPyBandits master branch. To easily update SMPyBandits: `pip install git+git://github.com/SMPyBandits/SMPyBandits.git --upgrade`

2. Search for similar issues. Make sure to delete `is:open` on the issue search to find solved tickets as well. It's possible somebody has encountered this bug already. Still having a problem? Open an issue on Github to let us know.

3. Make sure you provide us with useful information about your configuration: what OS are you using? Which version of Python? (3, [no one should use Python 2](https://pythonclock.org/) anymore...) etc

4. Provide us with a script to reproduce the issue. This script should be runnable as-is. I recommend that you use Github Gists to post your code. Any issue that cannot be reproduced is likely to be closed.

5. If possible, take a stab at fixing the bug yourself --if you can!

The more information you provide, the easier it is for us to validate that there is a bug and the faster we'll be able to take action. If you want your issue to be resolved quickly, following the steps above is crucial.

---

## Requesting a Feature

You can also use Github issues to request features you would like to see in SMPyBandits, or changes in the SMPyBandits API.

1. Provide a clear and detailed explanation of the feature you want and why it's important to add. Keep in mind that I want features that will be useful to my research. Any other requests for features will probably stay unanswered.

2. Provide code snippets demonstrating the API you have in mind and illustrating the use cases of your feature (eg. for a new algorithm, give a reference to the research article). Of course, you don't need to write any real code at this point!

3. After discussing the feature you may choose to attempt a Pull Request. If you're at all able, start writing some code. I always have more work to do than time to do it. If you can write some code then that will speed the process along.


---

## Requests for Contributions

[This is the file](https://github.com/SMPyBandits/SMPyBandits/tree/master/TODO.md) where I list current issues and features to be added. If you want to start contributing to SMPyBandits, this is the place to start.

---

## Pull Requests

**Where should I submit my pull request?**

1. **SMPyBandits improvements and bugfixes** go to the [SMPyBandits `master` branch](https://github.com/SMPyBandits/SMPyBandits/tree/master).

Please note that PRs that are primarily about **code style** (as opposed to fixing bugs, improving docs, or adding new functionality) will likely be rejected.

Here's a quick guide to submitting your improvements:

1. If your PR introduces a change in functionality, make sure you start by writing a design doc and share it on a SMPyBandits issue.

2. Write the code (or get others to write it). This is the hard part!

3. Make sure any new function or class you introduce has proper docstrings. Make sure any code you touch still has up-to-date docstrings and documentation. I don't care about docstring style, just useful docstrings.

4. Write examples, and try your code in some non-trivial bandit problems.

5. I use PEP8 syntax conventions, but I am not dogmatic when it comes to line length. Make sure your lines stay reasonably sized, though. To make your life easier, I recommend running a PEP8 linter:
    - Install PEP8 packages: `pip install pep8 pytest-pep8 autopep8`
    - Run a standalone PEP8 check: `py.test --pep8 -m pep8`
    - You can automatically fix some PEP8 error by running: `autopep8 -i --select <errors> <FILENAME>` for example: `autopep8 -i --select E128 Policies/newAlgorithm.py`

6. When committing, use appropriate, descriptive commit messages.

7. Update the documentation. If introducing new functionality, make sure you include code snippets demonstrating the usage of your new feature.

8. Submit your PR. If your changes have been approved in a previous discussion, and if you have complete (and passing) unit tests as well as proper docstrings/documentation, your PR is likely to be merged promptly.

---

## Adding new examples

Even if you don't contribute to the SMPyBandits source code, if you have an application of SMPyBandits that is concise and powerful, please consider adding it to my collection of examples, as a notebook.
[Existing notebooks](https://github.com/SMPyBandits/SMPyBandits/tree/master/notebooks) show idiomatic SMPyBandits code: make sure to keep your own script in the same spirit.
