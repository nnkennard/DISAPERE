// Radio buttons

$(document).ready(function() {
    $('input:radio[name=review_view]').change(function() {
        if (this.value == 'para') {
            document.getElementById('review-paragraph-browser').setAttribute("style", "display:inline")
            document.getElementById('review-table-browser').setAttribute("style", "display:none")
        }
        else if (this.value == 'table') {
            document.getElementById('review-paragraph-browser').setAttribute("style", "display:none")
            document.getElementById('review-table-browser').setAttribute("style", "display:inline")
        }
    });
    $('input:radio[name=rebuttal_view]').change(function() {
        if (this.value == 'para') {
            document.getElementById('rebuttal-paragraph-browser').setAttribute("style", "display:inline")
            document.getElementById('rebuttal-table-browser').setAttribute("style", "display:none")
        }
        else if (this.value == 'table') {
            document.getElementById('rebuttal-paragraph-browser').setAttribute("style", "display:none")
            document.getElementById('rebuttal-table-browser').setAttribute("style", "display:inline")
        }
    });
});



// tabs
const tabs = document.querySelectorAll('.tabs li');
const tabContentBoxes = document.querySelectorAll('#tab-content > div');

tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        tabs.forEach(item => item.classList.remove('is-active'));
        tab.classList.add('is-active');

        const target = tab.dataset.target;
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
rebuttal_label_keys = ["coarse", "fine"]

function clear_labels(){
  for (key of review_label_keys) {
        document.getElementById("rev_para_" + key).innerHTML = ""
    }
  for (key of rebuttal_label_keys) {
        document.getElementById("reb_para_" + key).innerHTML = ""
    }
  const context_sentences = document.querySelectorAll('.contsent');
  for (sent of context_sentences){
    sent.removeAttribute('highlighted')
  }
}

function populate_review_labels(element) {
    relevant = example_data["review_sentences"][element.getAttribute("sentence_index")]
    for (key of review_label_keys) {
        document.getElementById("rev_para_" + key).innerHTML = relevant[key]
    }
}

function populate_rebuttal_labels(element) {
    relevant = example_data["rebuttal_sentences"][element.getAttribute("sentence_index")]
    for (key of rebuttal_label_keys) {
        document.getElementById("reb_para_" + key).innerHTML = relevant[key]
    }
    maybe_alignments = relevant["alignment"][1]
    if ( maybe_alignments != null)    {
        for (aligned_index of maybe_alignments){
          for (sent of document.querySelectorAll('.contsent')){
            if (sent.getAttribute("sentence_index") == aligned_index.toString()) {
              sent.setAttribute('highlighted', "yes")

            }
          }
        }
    }
}

prepare_review_display()
prepare_rebuttal_display()

function prepare_review_display() {

    review_area = document.getElementById("review_paragraph_area");

    sentences = []
    for (review_sentence_index in example_data["review_sentences"]) {
        review_sentence = example_data["review_sentences"][review_sentence_index]
        sentence_text = review_sentence["text"]
        suffix = review_sentence["suffix"].replaceAll("\n", "<br/>")
        review_area.innerHTML += "<span class=\"sentence\" sentence_index=" + review_sentence_index + " onMouseout=\"clear_labels()\" onMouseover=\"populate_review_labels(this)\">" + sentence_text + "</span>" + suffix
    }

    table = document.getElementById("review_table")
    table.innerHTML = "<tr><th>Sentence</th><th>Review action</th><th>Aspect</th><th>Polarity</th><th>Fine review action</th></tr>"
    for (sentence of example_data["review_sentences"]) {
        row = table.insertRow()
        sentence_cell = row.insertCell()
        sentence_cell.innerHTML = sentence["text"] + sentence["suffix"]
        for (key of review_label_keys) {
            cell = row.insertCell()
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

function prepare_rebuttal_display(){
  review_area = document.getElementById("review_in_rebuttal_paragraph_area");
  review_area.innerHTML = ""

  sentences = []
  for (sentence_index in example_data["review_sentences"]) {
      review_sentence = example_data["review_sentences"][sentence_index]
      sentence_text = review_sentence["text"]
      suffix = review_sentence["suffix"].replaceAll("\n", "<br/>")
      review_area.innerHTML += "<span class=\"contsent\" sentence_index=" + sentence_index + ">" + sentence_text + "</span>" + suffix
  }


  review_area = document.getElementById("review_in_rebuttal_table_area");
  review_area.innerHTML = ""

  sentences = []
  for (sentence_index in example_data["review_sentences"]) {
      review_sentence = example_data["review_sentences"][sentence_index]
      sentence_text = review_sentence["text"]
      suffix = review_sentence["suffix"].replaceAll("\n", "<br/>")
      review_area.innerHTML += "<span class=\"contsent\" sentence_index=" + sentence_index + ">" + sentence_text + "</span>" + suffix
  }




  rebuttal_area = document.getElementById("rebuttal_paragraph_area");

  sentences = []
  for (sentence_index in example_data["rebuttal_sentences"]) {
      rebuttal_sentence = example_data["rebuttal_sentences"][sentence_index]
      sentence_text = rebuttal_sentence["text"]
      suffix = rebuttal_sentence["suffix"].replaceAll("\n", "<br/>")
      rebuttal_area.innerHTML += "<span class=\"sentence\" sentence_index=" + sentence_index + " onMouseout=\"clear_labels()\" onMouseover=\"populate_rebuttal_labels(this)\">" + sentence_text + "</span>" + suffix
  }

  table = document.getElementById("rebuttal_table")
    table.innerHTML = "<tr><th>Sentence</th><th>Rebuttal stance</th><th>Rebuttal action</th></tr>"
    for (sentence_index in example_data["rebuttal_sentences"]) {
        sentence = example_data["rebuttal_sentences"][sentence_index]
        row = table.insertRow()
        row.setAttribute('sentence_index', sentence_index)
        row.setAttribute('onMouseover', "populate_rebuttal_labels(this)")
        row.setAttribute('onMouseout', "clear_labels()")
        sentence_cell = row.insertCell()
        sentence_cell.innerHTML = sentence["text"] + sentence["suffix"]
        for (key of rebuttal_label_keys) {
            cell = row.insertCell()
            maybe_value = sentence[key]
            if (maybe_value == "none"){
              maybe_value = ""
            } else {
              maybe_value = maybe_value.split("_")[1]
            }

            cell.innerHTML = sentence[key]
        }
    }

}