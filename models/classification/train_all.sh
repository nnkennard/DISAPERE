for category in fine_review_action  polarity  rebuttal_action  rebuttal_stance  review_action aspect
do
	python train_eval.py -m train -c $category -d prepared_data
	python train_eval.py -m eval -c $category -d prepared_data >> results.txt
done
