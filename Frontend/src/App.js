import React from 'react';
import 'antd/dist/antd.css';
import BaseRouter from "./routes";
import {BrowserRouter as Router} from "react-router-dom";
import CustomLayout from "./containers/Layout";
import {connect} from "react-redux";
import  * as actions from "./store/actions/authActions";
class App extends React.Component {
  componentDidMount() {
    this.props.onTryAutoSignUp();
  }
  render(){
      return (
      <div className="App">
        <Router>
          <CustomLayout {...this.props}>
            <BaseRouter />
          </CustomLayout>
        </Router>
      </div>
    );
  }
  
}
const mapStateToProps = state => {
    return {
      isAuthenticated: state.token !== null ,
      username: state.username
    }
}

const mapDispatchToProps = dispatch => {
  return {
    onTryAutoSignUp: () =>  dispatch(actions.authCheckState)
  }
}

export default connect(mapStateToProps, mapDispatchToProps)(App);
