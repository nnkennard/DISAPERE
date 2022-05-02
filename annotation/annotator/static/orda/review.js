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


document.getElementById("submitBtn").disabled = "true";
document.getElementById("tok_merge_prev_0").disabled = "true";


var num_review_sentences = getJsonified("num_review_sentences");
var metadata = getJsonified("metadata");

var merge_prev = new Array(num_review_sentences).fill(0);


function initialize(){
    for (var i in Array(num_review_sentences).fill(0)){
        document.getElementById("tok_merge_prev_" + i.toString()).checked = null
    }
}


function addArg(clickedButton) {
    var sentence_index = last(clickedButton.id.split("-"));
    document.getElementById("arg-1-" + sentence_index).style = "display:block";
    clickedButton.style = "display:none";

}

function revealItem(original_dropdown, key) {
    document.getElementById(original_dropdown.id.replace("arg-d", key)).style = "display:block";
}

function concealItem(original_dropdown, key) {
    document.getElementById(original_dropdown.id.replace("arg-d", key)).style = "display:none";
}


var menu_map = getJsonified("menu_map");
var all_menus = ["arg", "asp", "pol", "req", "struc"];
var full_menu_name_map = {
    "arg": "Argument",
    "asp": "Aspect",
    "pol": "Polarity",
    "req": "Request type",
    "struc": "Structuring type"
};


function argChange(argDropdown) {
    var val = argDropdown.value;
    if (val == "Other") {
        return;
    }
    for (var menu of all_menus) {
        if (menu == "arg") {
            continue;
        }
        if (menu_map[val].required.includes(menu) || menu_map[val].allowed.includes(menu)) {
            revealItem(argDropdown, menu);
        } else {
            concealItem(argDropdown, menu);
        }
    }
}


function tokenErrorClicked(clickedButton) {
    var sentence_index = last(clickedButton.id.split("_"));
    for (var bla of["0", "1"]) {
        for (var menu of all_menus) {
            var element = document.getElementById(menu + "-d-" + bla + "-" + sentence_index);
            if (clickedButton.checked) {
                element.disabled = "true";
            } else {
                element.removeAttribute("disabled");
            }
        }
    }
    var argBtn = document.getElementById("addArgBtn-" + sentence_index);
    if (clickedButton.checked) {
        argBtn.disabled = "true";
        merge_prev[parseInt(sentence_index)] = 1;
    } else {
        argBtn.removeAttribute("disabled");
        merge_prev[parseInt(sentence_index)] = 0;
    }
}

function buildAnnotations(all_labels) {
    var is_valid = !(document.getElementById("egregious_tok").checked);
    var review_builder = {
        "review_annotation": {

            "review_id": metadata.review_id,
            "overall_comment": document.getElementById("comments").value,
            "is_valid": is_valid,
            "errors": JSON.stringify({
                "egregious_tokenization": document.getElementById("egregious_tok").checked,
                "merge_prev": merge_prev,
            }),
            "initials": metadata.initials,
            "time_to_annotate": getElapsedTime(),
            "start_time": start_time,

        },
        "review_sentence_annotations": []

    };
    if (!is_valid) {
        return review_builder;
    } else {
        for (var sentence_index in all_labels) {
            var sentence_label = all_labels[sentence_index];
            review_builder.review_sentence_annotations.push({

                "review_id": metadata.review_id,
                "review_sentence_index": sentence_index,
                "initials": metadata.initials,
                "labels": JSON.stringify(sentence_label)

            });

        }
    }
    if (review_builder.review_sentence_annotations.length < num_review_sentences){
      console.log("Problem -- too few sentences")
      console.log(review_builder.review_sentence_annotations)
    }
    return review_builder;
}

function getDropdownName(label, arg_num, sentence_index) {
    return label + "-d-" + arg_num + "-" + sentence_index.toString();
}



function getDropdown(label, arg_num, sentence_index) {
    return document.getElementById(getDropdownName(label, arg_num, sentence_index));
}


function isDropdownSet(label, arg_num, sentence_index) {
    console.log("Checking dropdown", label, arg_num, sentence_index)
    return !(getDropdown(label, arg_num, sentence_index).value.startsWith("--"));
}


function validateArgument(sentence_index) {
    var labels = {
        "0": {},
        "1": {}
    };

    if (!isDropdownSet("arg", "0", sentence_index)){
        alert("Please select an argument for sentence " + sentence_index + ".");
        return null;
    }

    for (var arg_num of ["0", "1"]){
        if (isDropdownSet("arg", arg_num, sentence_index)){
            var arg_value = getDropdown("arg", arg_num, sentence_index).value;
            console.log("Considering arg", arg_value)
            for (var menu of menu_map[arg_value].required) {
                if (!isDropdownSet(menu, arg_num, sentence_index)) {
                    alert("Please add " + full_menu_name_map[menu] + " for sentence " + sentence_index + "(argument " + arg_num + ")");
                    return null;
                }
            }
            for (var menu_to_label of ["arg"].concat(menu_map[arg_value].required, menu_map[arg_value].allowed)) {
                console.log("Getting label:", menu_to_label)
                if (isDropdownSet(menu_to_label, arg_num, sentence_index)) {
                    labels[arg_num][menu_to_label] = getDropdown(menu_to_label, arg_num, sentence_index).value;
            }
        }

    }
}
    return labels;
}



function validateAll() {
    var all_labels = Array();

    for (var i of Array(num_review_sentences).keys()) {

        if (merge_prev[i]) {
                all_labels.push({});
            // Nothing to check
        } else if (!isDropdownSet("arg", "0", i)) {
            alert("Please enter an argument value for sentence " + i.toString());
            return;
        } else {
            var maybe_labels = validateArgument(i);
            if (maybe_labels === null) {
                return;
            } else {
                all_labels.push(maybe_labels);
            }
        }
    }

    var result = buildAnnotations(all_labels);

    document.getElementById("id_annotation").value = JSON.stringify(result);
    document.getElementById("submitBtn").removeAttribute("disabled");
}
