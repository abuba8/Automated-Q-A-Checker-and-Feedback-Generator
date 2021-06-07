import * as actionTypes from '../actions/actionTypes';
import { update } from '../utility';
const initialState = {
    token: null,
    username: null,
    signup: null,
    error: null, 
    loading: false
}
const authStart = (state, action) => {
    return update(state, {
        error: null,
        loading: true
    });
}
const authSuccess = (state, action) => {
    console.group(action.username)
    return update(state, {
        token: action.token,
        username: action.username,
        signup: action.signup,
        error: null,
        loading: false
    });
}
const authFail = (state, action) => {
    return update(state, {
        error: action.error,
        loading: false
    });
    window.location = "/login";
}
const authLogout = (state, action) => {
    return update(state, {
        token: null
    });
}
const reducer = (state=initialState, action) => {
    switch (action.type) {
        case actionTypes.AUTH_START: return authStart(state, action);
        case actionTypes.AUTH_SUCCESS: return authSuccess(state, action);
        case actionTypes.AUTH_FAIL: return authFail(state, action);
        case actionTypes.AUTH_LOGOUT: return authLogout(state, action);
        default:
            return state;
    }
}
export default reducer;