from models.abstract_model import Model
import numpy as np
from collections import defaultdict
from datetime import datetime
from dataset import read_csv
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

class LR_Model(Model):
    '''logistics regression
    '''
    def __init__(self, config) -> None:
        super().__init__(config)
        self.pre_process_features()
        self.featured_X = np.array([self.feature(d[0], d[1]) for d in self.train_X])
        self.RL_model  = Ridge()
        self.scaler = StandardScaler()
        
    def predict(self, datum):
        user, recipe,  date, review =  datum
        if user not in self.user_feature_vector or recipe not in self.recipe_feature_vector:
            pred = 5
        else:
            pred = self.RL_model.predict(self.scaler.transform([self.feature(user, recipe)]))[0]
        return pred
    
    def train(self):
        
        self.RL_model.fit(self.scaler.fit_transform(self.featured_X),self.train_Y)

    
    def load_best_params(self):
        return self
    
    
    
    def feature(self, user, recipe):
        return  self.user_feature_vector[user] + self.recipe_feature_vector[recipe]
    
    
    
    def pre_process_features(self):
        print("pre-processing features")
        # construct user-item map (of training)
        interactionsPerUser = defaultdict(list)
        interactionsPerRecipe = defaultdict(list)

        for (user, recipe, date, review), rating in  zip(self.train_X, self.train_Y):
            rating = int(rating)
            interactionsPerUser[user].append((recipe,rating,date,review))
            interactionsPerRecipe[recipe].append((user,rating,date,review))
            
        interactions = self.interations
        
        ################
        # User Feature 
        ################     
        self.user_feature_vector = {}
        # Rating distribution (6)
        # Rating variance (1)
        # Number of reviews (1)
        # Time of first review (3)
        # Time of last review (3)
        # Average review length (normalized) (1)
        # Review frequency (per day) (1)

        max_review_length = max([len(i[4]) for i in interactions])
        # User Feature Per User
        for user in interactionsPerUser.keys():
            user_interactions = interactionsPerUser[user]
            features = [0] * 16

            for recipe,rating,date,review in user_interactions:
                rating = int(rating)
                features[rating] += 1

            all_ratings = [int(i[1]) for i in user_interactions]
            variance = np.var(all_ratings)
            features[6] = variance

            features[7] = len(user_interactions)

            all_dates = [datetime.strptime(i[2], '%Y-%m-%d') for i in user_interactions]
            all_dates.sort()
            # Time of first review
            features[8] = all_dates[0].year
            features[9] = all_dates[0].month
            features[10] = all_dates[0].day
            # Time of last review
            features[11] = all_dates[-1].year
            features[12] = all_dates[-1].month
            features[13] = all_dates[-1].day

            all_reviews = [i[3] for i in user_interactions]
            avg_review_length = sum([len(r) for r in all_reviews]) / len(all_reviews)
            features[14] = avg_review_length / max_review_length

            day_diff = (all_dates[-1] - all_dates[0]).days
            avg_review_per_day = len(all_reviews) / day_diff if day_diff else 0
            features[15] = avg_review_per_day

            self.user_feature_vector[user] = features
            
        ################
        # Recipe Feature 
        ################
        recipe_header, recipes = read_csv('archive/RAW_recipes.csv')
        self.recipe_feature_vector = {}
            
        # minutes (1)
        # submitted (3)
        # tag count (1)
        # n_steps (1) 
        # steps length (1)
        # description length (1)
        # n_ingredients (1)

        max_step_length = max([len(r[8]) for r in recipes])
        max_desc_lenth = max([len(r[9]) for r in recipes])

        for r in recipes:
            recipe_id = r[1]
            features = [0] * 9

            minutes = int(r[2])
            submitted = datetime.strptime(r[4], '%Y-%m-%d')
            n_tag = len(eval(r[5]))
            n_steps = int(r[7])
            step_len = len(r[8])
            desc_len = len(r[9])
            n_ingredients = int(r[11])

            features[0] = minutes
            features[1] = submitted.year
            features[2] = submitted.month
            features[3] = submitted.day
            features[4] = n_tag
            features[5] = n_steps
            features[6] = step_len / max_step_length
            features[7] = desc_len / max_desc_lenth
            features[8] = n_ingredients

            self.recipe_feature_vector[recipe_id] = features
    
                