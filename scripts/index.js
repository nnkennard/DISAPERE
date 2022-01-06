// tabs
const tabs = document.querySelectorAll('.tabs li');
const tabContentBoxes = document.querySelectorAll('#tab-content > div');

tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        tabs.forEach(item => item.classList.remove('is-active'));
        tab.classList.add('is-active');

        const target = tab.dataset.target;
        // console.log(target);
        tabContentBoxes.forEach(box => {
            if (box.getAttribute('id') === target) {
                box.classList.remove('is-hidden');
            } else {
                box.classList.add('is-hidden');
            }
        })
    })
})


review_label_keys = ["coarse", "asp", "pol", "fine"]

function populate_review_labels(element) {
    relevant = example_data["review_sentences"][element.getAttribute("sentence_index")]
    for (key of review_label_keys) {
        document.getElementById("rev_para_" + key).innerHTML = relevant[key]
    }
}

prepare_review_display()

function prepare_review_display() {

    review_area = document.getElementById("review_paragraph_area");

    sentences = []
    for (review_sentence_index in example_data["review_sentences"]) {
        review_sentence = example_data["review_sentences"][review_sentence_index]
        sentence_text = review_sentence["text"]
        suffix = review_sentence["suffix"].replaceAll("\n", "<br/>")
        review_area.innerHTML += "<span class=\"sentence\" sentence_index=" + review_sentence_index + " onMouseover=\"populate_review_labels(this)\">" + sentence_text + "</span>" + suffix
    }


    table = document.getElementById("review_table")
    table.innerHTML = "<th>Sentence</th><th>Review action</th><th>Aspect</th><th>Polarity</th><th>Fine review action</th>"
    for (sentence of example_data["review_sentences"]) {
        row = table.insertRow()
        sentence_cell = row.insertCell()
        sentence_cell.innerHTML = sentence["text"] + sentence["suffix"]
        for (key of review_label_keys) {
            cell = row.insertCell()
            console.log(sentence[key],sentence[key].split("_")[1])
            maybe_value = sentence[key]
            if (maybe_value == "none"){
              maybe_value = ""
            } else {
              maybe_value = maybe_value.split("_")[1]
            }

            cell.innerHTML = maybe_value
        }
    }
}