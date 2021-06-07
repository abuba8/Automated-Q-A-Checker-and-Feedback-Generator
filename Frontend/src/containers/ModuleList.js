import React from "react";
import Modules from "../component/Modules";
import axios from "axios";
import "../css/ModuleList.css"
import {Button} from "antd";
import {Link} from "react-router-dom";
class ModuleList extends React.Component {
    state = {
        modules: []
    }
    componentDidMount()
    {
        axios.get('http://127.0.0.1:8000/api/modules')
        .then(res =>{ 
            this.setState({
                modules: res.data
            })
        })
        console.log(this.state.modules)
    }
    handleClick = () => { 
        this.setState({
            entermodule: true,
        })
    }
    render()
    {
        return(
            <div>
                <Modules data={this.state.modules} user={this.props.match.params.id}/>
                <br />
                <div style={{display: 'flex', justifyContent: 'center'}}>
                <Link to={'/addmodule' + '/' + this.props.match.params.id}>
                    <Button>Enter a new module !</Button>
                </Link>
                </div>
            </div>    
        )
    }
}
export default ModuleList;