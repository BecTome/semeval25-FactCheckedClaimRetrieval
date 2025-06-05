# AI-powered Claim matching

## Claim matching
We understand claim matching as the task of identifying statements that share a
common meaning, even if they are expressed in different ways [(Larraz, I., Míguez, R. and
Salitelli, F., 2023)](https://revista.profesionaldelainformacion.com/index.php/EPI/article/view/87284/63433)

More specifically to our needs, it is the process of identifying pairs of textual messages
containing claims that can be served with one fact-check. In this scenario one of the claims
in the pair is unverified while the other one has already been verified by a human fact-checker.

## Task definition
Given a dataset with (3868) similarity (or dissimilarity) relations between:
* a *unverified claim* (or search terms with potential to be verified),
* a *reviewed claim* (by human fact-checkers), with the text supporting the verification (evidences)
and other "metadata" or associated resources (e.g. summary, description, keywords, ...)

... with the *similarity relation* manually annotated by experts using contextual
information from a claim review or fact-check

The objective is to build an AI model or agent able to find semantic similarity matches between one unverified
claim and verified claims (supported by evidence) in a multilingual context. This challenge is a
classification and ranking problem, where your solution will be asked to produce a ranked list of top-n
candidates similar to such unverified claim.

We will evaluate the solution using a dataset with similar characteristics to the one provided, but
the important part will be the review of the ideas contributed to the solution of the problem and its
implementation.

There are no limitations to using NLI models, LLMs or any framework as long as it is justified to develop
the proposal.

### Requirements
* Python as programming language and Jupyter Notebook as runtime environment
* You should propose an evaluation metric and/or evaluation frameworks
* We expect your solution to use the claim text but also the other information in the claim review article 
and/or external information for the improvement of results


## The Data
The attached dataset contains unverified claims that have been labelled as similar (or dissimilar) to
verified/reviewed claims available in our claim review database. Also includes contextual information
about these verifications/reviews, such as full text, summary, url, description, keywords, dates or
attached multimedia resources (image, video).


### Considerations
* Each row represents a similarity relation (established by humans) between an unverified claim and a
verified/reviewed claim (1 match; 0 no match)
* Each claim review is univocally represented by its url. Nevertheless, several claim reviews could
debunk the same claim
* Also, each verified/reviewed claim could be related to many unverified claims and viceversa


### Field description

#### Basic data
* **unverified claim** *(str, multilingual (en, es))*: unverified claim / search term
* **reviewed claim** *(str, multilingual)*: reviewed claim / title of the claim review article
* **similarity** *(int)*: label indicating a positive (1) or negative (0) similarity between 

#### Claim review article
* **title** *(str, multilingual)*: article title
* **text** *(str, multilingual)*: original text. It can contains "noise" in the sense of including 
some incorrect paragraphs (texts of menus, whitespaces, symbols, etc.) 
* **meta_description** *(str, multilingual)*: description extracted from meta tags and/or json+ld representation
* **summary** *(str, multilingual)*: summary generated from original text (using an external service)
* **kb_keywords** *(list[str], multilingual)*: keywords generated from original text (ngrams range 1-3) using KeyBert
* **meta_keywords** *(list[str], multilingual)*: keywords extracted from meta tags and/or json+ld representation
* **url** *(string, url)*: url of the claim review
* **domain** *(string, url)*: domain (of the publisher)
* **published** *(datetime)*: publication date
* **cm_authors** *(list[str], multilingual)*: a list of authors (extracted from meta tags and/or json+ld representation)
* **cr_author_name** *(str): name of the author of the claim review (the fact-checker)
* **cr_country** *(string)*: country of publication
* **meta_lang** *(string, ISO Code)* native language of the claim review extracted from meta tags


#### Linked resources:
* **cr_image** *(str, url)*: (if any) image attached into the claim review
* **meta_image** *(list[str], url)*: (if any) linked images extracted from the meta and/or json+ld object
* **movies** *(list[str], url)*: (if any) videos linked on the meta and/or json+ld object


# References
[(Larraz, I., Míguez, R. and Salitelli, F., 2023). “Semantic similarity models for automated fact-checking:
ClaimCheck as a claim matching tool”. Profesional de la información, v. 32, n. 3, e320321.https://doi.org/10.3145/epi.2023.may.21]
(https://revista.profesionaldelainformacion.com/index.php/EPI/article/view/87284/63433)