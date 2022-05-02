// ==== Common, needs to be factored out
const start_time = Date.now();

function getElapsedTime() {
    return Math.floor(Date.now() - start_time / 1000);
}

function getJsonified(label) {
    return JSON.parse(document.getElementById(label).textContent);
}

function last(l) {
    return l[l.length - 1];
}

function sum(l) {
    return l.reduce(function(a, b) {
        return parseInt(a) + parseInt(b);
    }, 0)

}

// ==== End common stuff


STATUSES = ["NOT_STARTED", "STARTED", "VALIDATED"] // Change to not started if clicked sentence clicked any radio button, entered text

CURRENT_STATUS = "NOT_STARTED"


// Initialization

review_sentences = getJsonified("review_sentences");
other_annotations = getJsonified("other_annotations");
metadata = getJsonified("metadata");

document.getElementById("submitBtn").disabled = "true"
document.getElementById("nav:" + metadata["rebuttal_index"].toString()).classList.add("is-focused")
var highlighted = new Array(review_sentences.length).fill(0);
maybeDisableCopyPrevious()

function maybeDisableCopyPrevious() {
    if (metadata.rebuttal_index == 0 || other_annotations.statuses[metadata.rebuttal_index - 1] == "Incomplete") {
        document.getElementById("copyBtn").disabled = true;
    }
    for (i in other_annotations.statuses) {
        if (other_annotations.statuses[i]) {
            document.getElementById("nav:" + i.toString()).classList.add("is-success")
        }
    }
}

function someHighlighted() {
    return sum(highlighted) > 0
}


function toggleModal() {
    modal = document.getElementById("rebuttalmodal")
    if (modal.classList.contains("is-active")) {
        modal.classList.remove("is-active")
    } else {
        modal.classList.add("is-active")
    }
}

function navSwitch(button_element) {
    new_index = last(button_element.id.split(":"))
    parts = window.location.href.split("/")
    new_url = parts.slice(0, 7).join("/") + "/" + new_index + "/" + parts.slice(8)
    window.location.replace(new_url)
}


// Validation


function isSentencesAreAlignedChecked() {
    return document.getElementById("alignment_categories").value == "Aligned sentences have been highlighted"
}

function checkNeedsExplanationButDoesntHave() {
    if (document.getElementById("comments").value.trim() == 0) {
        if (document.getElementById("label:radios:other").checked) {
            window.alert("Please specify why none of the label options apply")
            return 0
        } else if (document.getElementById("label:radios:multiple").checked) {
            window.alert("Please specify which multiple label options apply")
            return 0
        }
    }
    return 1
}

function checkNeedsSubtypeButDoesntHave(label) {
    if (label.startsWith("done") || label.startsWith("by-cr") || label.startsWith("reject-request") ){
        if (!label.includes("_")){
            window.alert("Please select the subtype for the selected relation label")
            return 0
        }
    }
    return 1
}

function getLabel() {
    labels = {}
    for (radio_button of document.getElementsByClassName("radio")) {
        if (radio_button.checked) {
            labels[radio_button.name] = radio_button.value;
        }
    }
    console.log(labels)
    if ("label:radios" in labels) {
        label = labels["label:radios"]
        for (var subtype of ["manu", "scope"]){
            if (subtype + "_" + label in labels) {
                label = label + "_" + subtype + "_" + labels[subtype + "_" + label]
            }
        }
        return label
    } else {
        return null;
    }
}

function validateAll() {

    result_builder = {

            "review_id": metadata.review_id,
            "rebuttal_id": metadata.rebuttal_id,
            "rebuttal_sentence_index": metadata.rebuttal_index,
            "initials": metadata.initials,
            "is_valid": !(document.getElementById("egregious_tok").checked),
            "comment": document.getElementById("comments").value,
            "errors": {
                "egregious_tok": document.getElementById("egregious_tok").checked,
                "merge_prev": document.getElementById("tok_merge_prev").checked
            },
            "time_to_annotate": getElapsedTime(),
            "start_time": start_time,

        }

    relation_label = getLabel();
    if (document.getElementById("egregious_tok").checked) {
        result_builder.aligned_review_sentences = []
        result_builder.relation_label = ""
        result_builder.alignment_category = ""

    } else {

    if (relation_label === null) {
        window.alert("Please select a relation label. If none applies, please select 'Other'.")
        return
    }
    if (isSentencesAreAlignedChecked() && !someHighlighted()) {
        window.alert("You have indicated that sentences are highlighted, but none are highlighted. Please fix.")
    } else if (!isSentencesAreAlignedChecked() && someHighlighted()) {
        window.alert("You have highlighted sentences, but selected a 'no context' option as well. Please fix.")
    } else if (!checkNeedsExplanationButDoesntHave()) {
        return
    } else if (!checkNeedsSubtypeButDoesntHave(relation_label)) {
        return
    } 
    else {
        result_builder.aligned_review_sentences = highlighted;
        console.log("bababa", relation_label)
        result_builder.relation_label = relation_label;
        result_builder.alignment_category = document.getElementById("alignment_categories").value;
    }
}
    document.getElementById("submitBtn").removeAttribute("disabled");
    document.getElementById("id_annotation").value = JSON.stringify(result_builder)
    CURRENT_STATUS = "VALIDATED"
}

TAG_TYPE_MAP = {
    "arg": "success",
    "asp": "link",
    "pol": "danger",
    "fine": "primary"
}

// Label propagation

function getReviewLabelTags(review_index) {

    labels = JSON.parse(other_annotations.review_annotations[review_index])
    tags = ""
    console.log(other_annotations.review_annotations[review_index])
    for (var arg_num of["0", "1"]) {
        label_map = labels[arg_num]
        for (var category in label_map) {
            label_value = label_map[category]
            tag_type = TAG_TYPE_MAP[category]
            tags += '<span class="tag is-' + tag_type + '">' + label_value + '</span>'
        }

    }

    return '<div class="tags are-small"> ' + tags + ' </div>'
}

function populatePreviewBox() {
    reviewtable = document.getElementById('reviewtable')
    reviewtable.innerHTML = ""

    for (i in highlighted) {
        if (highlighted[i]) {
            reviewtable.innerHTML += '<tr class="border_bottom"><td>' + document.getElementById(
                "sentence:" + i).innerText + ' </td> <td> ' + getReviewLabelTags(i) + ' </td> </tr>';
        }

    }
}


function copyPrevious() {
    rebuttal_index = metadata["rebuttal_sentence_index"]
    if (other_annotations.previous_alignment.length > 0) {
        highlighted = new Array(review_sentences.length).fill(0);
        old_highlighted = JSON.parse(other_annotations.previous_alignment)
        for (index in old_highlighted ) {
            if (old_highlighted[index]){
                highlighted[index] = 1;
            }
        }
        console.log(highlighted)
    } else {
        alert("No annotations available for the previous sentence.");
    }
    for (i in highlighted) {
        updateHighlight(i);
    }
    label = other_annotations.previous_label
    if (label.includes("_")){
        label = label.split("_")[0]
    }
    elementToChange = document.getElementById("label:radios:" + label)
    elementToChange.checked = true;
    relRadioChange(elementToChange)
    populatePreviewBox()
}

function standardTokenizationError() {
    if (!(document.getElementById("tok_merge_prev").checked)){
        return;
    }
    copyPrevious();
}

// Interaction management

function clicked(ele) {
    review_idx = parseInt(last(ele.id.split(":")));
    highlighted[review_idx] = 1 - highlighted[review_idx];
    updateHighlight(review_idx)
    CURRENT_STATUS = "STARTED"
    populatePreviewBox()
}

function updateHighlight(review_index) {
    sentence_element = document.getElementById("sentence:" + review_index)
    if (highlighted[review_index]) {
        sentence_element.style = "background-color:#d5f5e3"
    } else {
        sentence_element.style = ""
    }
}

function relRadioChange(changed_radio) {
    document.getElementById("label:select").value = changed_radio.value;
}

function relSelectChange(sel_element) {
    document.getElementById("label:radios:" + sel_element.value).checked = "true"
}

window.addEventListener('beforeunload', function(e) {
    if (CURRENT_STATUS == "STARTED") {
        e.preventDefault(); //per the standard
        e.returnValue = ''; //required for Chrome
    }
});