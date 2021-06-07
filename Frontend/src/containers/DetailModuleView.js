import React from "react";
import axios from "axios";
import  {Card, Button} from 'antd'
import CustomForm from "../component/Form";

class DetailModuleView extends React.Component {
    state = {
        module: {}
    }
    componentDidMount()
    {
        const moduleID = this.props.match.params.moduleID;
        axios.get(`http://127.0.0.1:8000/api/module/${moduleID}`)
        .then(res => {
            this.setState({
                module: res.data
            })
            console.log(res.data)
        })
    }
    handleDelete = (event) => {
        const moduleID = this.props.match.params.moduleID;
        axios.delete(`http://127.0.0.1:8000/api/module/${moduleID}/delete`)
        .then(res => {
            alert("Your module has been deleted!");
            this.props.history.push('/savedtests/' + this.props.match.params.userid);
            this.forceUpdate();
        })
        .catch(console.log("Error"))
    }
    render()
    {
        return(
            <div>
               <Card title={this.state.module.title}>
                    <p>Module Code: {this.state.module.code}</p>
                    <p>Description: {this.state.module.description}</p>
                    <p>Number Of Students: {this.state.module.num_students} </p>
                </Card>
                <CustomForm requestMethod="put"
                            moduleID={this.props.match.params.moduleID}
                            btnText="Update a module Information!"/> 
                <form onSubmit={this.handleDelete}>
                    <Button type="danger" htmlType="submit">Delete</Button>
                </form>
            </div>
            
        )
    }
}
export default DetailModuleView;