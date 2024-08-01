#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 08:28:18 2024

@author: laurenshores
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import joblib
import pickle

# Load your predictive model
#model = joblib.load('model_rus_randomForest_Top3.pkl')
#df= pd.read_csv("src/assets/AmzingRceTeams_Data.csv")
with open("src/model_rus_randomForest_Top3.pkl", "rb") as f:
    model = pickle.load(f)




#--------------------------------------------------------
# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# Define the app layout
app.layout = html.Div([
    html.Div([
        html.Img(src='src/assets/amazing_race_logo2.png', 
                 style={'width': '1400px', 'height': '200px'}),
        html.H1('The Amazing Race Prediction App')
    ], style={'textAlign': 'center'}),
    
    # Subheading/Description
    html.Div([
        html.H3(['Ever wondered how you and a mate would perform on The Amazing Race?', html.Br(),
                 'Complete the fields below to use this app to predict the probability of a team making it to the Top 3 in the competition.']),
    ], style={'textAlign': 'center', 'padding': '10px'}),
    
    # Individual Contestant Inputs
    
    html.Div([
        html.Div([
            html.H3('Contestant 1'),
            html.Br(),
            dcc.Input(id='age_input1', type='number',min=18, max=85, step=1, placeholder='Age', style={'width': '100%'}),
            html.Br(),
            html.Br(),
            dcc.Dropdown(
                id='sex_input1',
                options=['Male', 'Female'],
                placeholder='Male'
                ),
            html.Br(),
            dcc.Dropdown(
                id='state_input1',
                options=['AL','AK','AZ','AR','AS','CA',	'CO','CT','DE','DC','FL','GA','GU',	'HI','ID','IL','IN','IA','KS','KY',	'LA','ME','MD','MA',	'MI',	'MN',	'MS',	'MO',	'MT',	'NE',	'NV',	'NH',	'NJ',	'NM',	'NY',	'NC',	'ND',	'MP',	'OH',	'OK',	'OR',	'PA',	'PR',	'RI',	'SC',	'SD',	'TN',	'TX',	'TT',	'UT',	'VT',	'VA',	'VI',	'WA',	'WV',	'WI',	'WY'],
                placeholder='AL'
            ),
            html.Br(),
            dcc.Input(id='occupation_input1', type='text', placeholder='Enter one occupation for contestant 1', style={'width': '100%'}),
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),

        #------------
        html.Div([
            html.H3('Contestant 2'),
            html.Br(),
            dcc.Input(id='age_input2', type='number', min=18, max=85, step=1, placeholder='Age', style={'width': '100%'}),
            html.Br(),
            html.Br(),
            dcc.Dropdown(
                id='sex_input2',
                options=['Male', 'Female'],
                placeholder='Male'
            ),
            html.Br(),
            dcc.Dropdown(
                id='state_input2',
                options=['AL','AK','AZ','AR','AS','CA',	'CO','CT','DE','DC','FL','GA','GU',	'HI','ID','IL','IN','IA','KS','KY',	'LA','ME','MD','MA',	'MI',	'MN',	'MS',	'MO',	'MT',	'NE',	'NV',	'NH',	'NJ',	'NM',	'NY',	'NC',	'ND',	'MP',	'OH',	'OK',	'OR',	'PA',	'PR',	'RI',	'SC',	'SD',	'TN',	'TX',	'TT',	'UT',	'VT',	'VA',	'VI',	'WA',	'WV',	'WI',	'WY'],
                placeholder='AL'
            ),
            html.Br(),
            dcc.Input(id='occupation_input2', type='text', placeholder='Enter one occupation for contestant 2', style={'width': '100%'}),
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'})
    ]),
    
    #-----------------------------------------------------------------
    # Whole page input section
    html.Div([
        html.H3('Team information', style={'textAlign': 'center', 'padding': '10px'}),
        dcc.Dropdown(
            id='team_demographics',
            options=[
                {'label': 'African American', 'value': 'AficanAm_team'},
                {'label': 'Asian American', 'value': 'AsianAm_team'},
                {'label': 'Hispanic', 'value': 'Hispanic_team'},
                {'label': 'Interracial', 'value': 'Interracial_team'},
                {'label': 'LGBT', 'value': 'LGBT_team'},
                {'label': 'Disability', 'value': 'Disabled_team'},
                {'label': 'None of the Above', 'value': 'None_of_these'}
            ],
            multi = True,
            placeholder='Select all that apply. For race and ethnicity options, both contestants need to check the box for these options to be relevant (unless interracial)'
        ),
        html.Br(),
        #html.Br(),
        dcc.Dropdown(
            id='team_relationship',
            options=[
                {'label': 'Family', 'value': 'Family_team'},
                {'label': 'Friends', 'value': 'Friend_team'},
                {'label': 'Couple or Former Couple', 'value': 'Couple_or_Ex_team'},
                {'label': 'Married', 'value': 'Married_team'},
                {'label': 'Strangers', 'value': 'Stranger_team'}
            ],
            placeholder='Select an option for relationship of team'
        ),
        html.Br(),
        html.Br(),
        html.Button('Submit', id='submit-button', n_clicks=0, style={'fontSize': '20px', 'padding': '15px 30px', 'margin': 'auto', 'display': 'block'})
            #'textAlign': 'center', 'marginTop': '20px'})
    ], style={'padding': '20px'}), #style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),

        #-------------
        
   html.Div(id='output-container', style={'textAlign': 'center', 'marginTop': '20px'}
                                          )
])

# Define the callback to update the output
@app.callback(
    Output('output-container', 'children'),
    Input('submit-button', 'n_clicks'),
    State('age_input1', 'value'),
    State('sex_input1', 'value'),
    State('state_input1', 'value'),
    State('occupation_input1', 'value'),
    State('age_input2', 'value'),
    State('sex_input2', 'value'),
    State('state_input2', 'value'),
    State('occupation_input2', 'value'),
    State('team_demographics', 'value'),
    State('team_relationship', 'value')
)



def update_output(n_clicks, age_input1, sex_input1, state_input1, occupation_input1,
                  age_input2, sex_input2, state_input2, occupation_input2,
                  team_demographics, team_relationship):
    
    myfeatures = ['Age_x', 'Age_y', 'LGBT_team', 'AficanAm_team', 'Friend_team',
       'Family_team', 'Couple_or_Ex_team', 'Married_team', 'Stranger_team',
       'Female_team', 'Male_team', 'Coed_team', 'AsianAm_team',
       'Disabled_team', 'Hispanic_team', 'Interracial_team',
       'RealityShow_team', 'Avg_Age', 'Age_Diff', 'Same_State']
             
    
    age_input1 = np.where(age_input1 is None, 18 , age_input1)
    age_input2 = np.where(age_input2 is None, 18 , age_input2)
    
    output1 = None
    if n_clicks > 0:
        # Prepare the input data for the model
        input_data = pd.DataFrame({
            'Age_x': [age_input1],
            'Age_y': [age_input2],
            
            
            'LGBT_team': np.where('LGBT_team' in team_demographics, [1] , [0]),
            'AficanAm_team': np.where('AficanAm_team' in team_demographics, [1] , [0]),
            'AsianAm_team': np.where('AsianAm_team' in team_demographics, [1] , [0]),
            'Hispanic_team': np.where('Hispanic_team' in team_demographics, [1] , [0]),
            'Interracial_team': np.where('Interracial_team' in team_demographics, [1] , [0]) ,
            'Disabled_team': np.where('Disabled_team' in team_demographics, [1] , [0]) ,
            'RealityShow_team': [0],
            
            'Friend_team': np.where( team_relationship == 'Friend_team', [1] , [0]) ,
            'Family_team': np.where( team_relationship == 'Family_team', [1] , [0]) ,
            'Couple_or_Ex_team': np.where( team_relationship == 'Couple_or_Ex_team', [1] , [0]) ,
            'Married_team': np.where( team_relationship == 'Married_team', [1] , [0]) ,
            'Stranger_team': np.where( team_relationship == 'Stranger_team', [1] , [0]) ,
            
            
            'Female_team': np.where( sex_input1 == sex_input2 == 'Female', [1], [0]),
            'Male_team': np.where( sex_input1 == sex_input2 == 'Male', [1], [0]),
            'Coed_team': np.where( sex_input1 != sex_input2, [1], [0]),
            'Same_State': np.where( state_input1 == state_input2, [1], [0] ),
            'Avg_Age': [ round((age_input1 + age_input2)/2)],
            'Age_Diff': [ abs(age_input1 - age_input2) ]
        })
        
        # to insure feature order is same as trained model
        input_data = input_data[myfeatures]
        print(input_data)
       
        
        # Predict using the loaded model
        prediction = model.predict(input_data)
        
        pred_prob = model.predict_proba(input_data)
        print('prediction: ' + str(prediction))
        print('pred prob: ' + str(pred_prob))
        
        res1 = np.where(prediction[0] == 1, 'Top 3 Team', 'Not a Top 3 Team')
        
        res = f'Model prediction: {res1}'
        res2 =f'Prediction probability: {pred_prob[0][1]}'
        
        output1 = html.Div([
        html.Span(res, style={'fontWeight': 'bold', 'fontSize': '30px', 'color': '#071166'}),
        html.Br(),
        html.Br(),
        html.Span(res2, style={'fontWeight': 'bold', 'fontSize': '30px', 'color': '#071166'})
    ])
        
        
        return output1
    return output1

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
    
    