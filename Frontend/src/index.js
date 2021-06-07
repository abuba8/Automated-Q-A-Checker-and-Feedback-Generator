import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';
import * as serviceWorker from './serviceWorker';
import reducer from "./store/reducers/auth";
import {createStore, compose, applyMiddleware} from "redux";
import {Provider} from "react-redux";
import thunk from "redux-thunk";
const composeEnhances = window.__REDUX_DEVTOOLS_EXTENSION_COMPOSE__ || compose;
const store = createStore(reducer, composeEnhances(
    applyMiddleware(thunk)
));
const main = (
    <Provider store = {store}>
        <App />
    </Provider>
)
ReactDOM.render(main, document.getElementById('root'));
serviceWorker.unregister();